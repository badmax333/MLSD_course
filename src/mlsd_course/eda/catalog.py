"""Синтетический каталог фич Альфа-Банка и размеченные запросы к нему.

ВНИМАНИЕ: это ДОПУЩЕНИЕ, а не данные банка. Каталог `data/synthetic/alfa_feature_catalog.csv`
написан вручную по публичному представлению о продуктах банка. Запросы взяты из публичных
датасетов (banking77_ru, synthetic-search-queries-ru) и шаблонов, метки 0/1/2 проставлены
правилами: целевая фича — 2, другая фича того же раздела — 1, остальные — 0.
Это ровно формат будущей ручной разметки банка «запрос → ожидаемая фича».
"""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from mlsd_course.eda.banking_groups import FAQ_INTENTS, SEARCH_INTENTS

CATALOG_PATH = (
    Path(__file__).resolve().parents[3] / "data" / "synthetic" / "alfa_feature_catalog.csv"
)

# Правила «запрос synthetic-search-queries-ru → фича каталога» (ДОПУЩЕНИЕ, ручные регулярки).
# Порядок важен: запросы про недвижимость без ипотеки считаем вне каталога (банк не продаёт
# квартиры), а «кредитная карта» не должна попадать в «кредит наличными».
OUT_OF_CATALOG = re.compile(
    r"недвижим|квартир|таунхаус|аренд|помещени|новострой|жил[ьоыи]|\bдом|участ|стартап|ipo"
    r"|сда[её]тся|сдам|сдаю"
)
SSQ_RULES: list[tuple[str, str]] = [
    (r"ипотек|ипотечн", "F40"),
    (r"кредитн\w* карт|кредитк", "F02"),
    (r"дебетов\w* карт", "F01"),
    (r"кэшб|кешб", "F07"),
    (r"автокредит|кредит\w*\W+(\w+\W+){0,3}(авто|машин)", "F38"),
    (r"рефинанс", "F39"),
    (r"\bкредит(?!н)|\bкредита|\bкредиты|займ|заём|заем", "F37"),
    (r"\bвклад|депозит", "F41"),
    (r"накопит|сбережен", "F42"),
    (r"пенси", "F43"),
    (r"\bиис\b|индивидуальн\w* инвестиционн", "F45"),
    (r"брокер|\bакци[йияюе]|облигац|фонд|инвестиц|портфел|вложени|актив", "F44"),
    (r"(страх\w*|полис)\W.*(поездк|путешеств|выезд|границ|туристическ|канад)", "F46"),
    (r"страх|полис", "F47"),
    (r"налог|вычет|деклараци|ндфл", "F48"),
    (
        r"выписк|истори\w* (операц|транзакц|покупок|платеж)|транзакц|операци\w* по (счет|карт)"
        r"|отч[её]т\w* по (счет|операц)",
        "F24",
    ),
    (r"банкомат|отделени\w* банк|офис\w* банк|банковск\w* (офис|отделени)", "F31"),
    (r"курс\w* (валют|доллар|евро)", "F27"),
    (r"обмен\w* валют|конверт", "F26"),
    (r"мобильн\w* (связ|телефон)|сч[её]т мобильн|жкх|коммунал", "F25"),
    (r"перевод|перевест|переведи", "F19"),
]

# Неверная раскладка: русский текст, набранный на английской раскладке.
_RU = "йцукенгшщзхъфывапролджэячсмитьбюё"
_EN = "qwertyuiop[]asdfghjkl;'zxcvbnm,.`"
LAYOUT = str.maketrans(_RU, _EN)

PRODUCT_SECTIONS = {"Карты", "Кредиты", "Ипотека", "Сбережения", "Инвестиции", "Страхование"}


def load_catalog(path: Path = CATALOG_PATH) -> pd.DataFrame:
    cat = pd.read_csv(path).fillna("")
    cat["doc"] = cat["title"] + ". " + cat["description"]
    return cat


def ssq_targets(text: str) -> list[str]:
    """Целевые фичи для запроса synthetic-search-queries-ru по правилам SSQ_RULES."""
    t = text.lower()
    if OUT_OF_CATALOG.search(t) and not re.search(r"ипотек", t):
        return []
    hits: list[str] = []
    for pattern, fid in SSQ_RULES:
        if re.search(pattern, t) and fid not in hits:
            if fid == "F37" and "F38" in hits:
                continue
            hits.append(fid)
    return hits


def _typo(text: str) -> str:
    """Детерминированная опечатка: две соседние буквы в середине самого длинного слова."""
    words = text.split()
    i = max(range(len(words)), key=lambda k: len(words[k]))
    w = words[i]
    if len(w) >= 4:
        m = len(w) // 2
        words[i] = w[: m - 1] + w[m] + w[m - 1] + w[m + 1 :]
    return " ".join(words)


def template_queries(cat: pd.DataFrame) -> pd.DataFrame:
    """Шаблонные короткие запросы по синонимам карточки: 4 страты на синоним."""
    rows = []
    for _, c in cat.iterrows():
        for syn in [s.strip() for s in c["synonyms"].split(";") if s.strip()]:
            variants = {
                "чистый": syn,
                "опечатка": _typo(syn),
                "раскладка": syn.translate(LAYOUT),
                "с брендом": (
                    f"оформить {syn} альфа"
                    if c["section"] in PRODUCT_SECTIONS
                    else f"{syn} альфа банк"
                ),
            }
            for stratum, q in variants.items():
                rows.append(
                    {
                        "text": q,
                        "source": "шаблон",
                        "stratum": stratum,
                        "targets": [c["feature_id"]],
                    }
                )
    return pd.DataFrame(rows)


def b77ru_queries(b77ru: pd.DataFrame, cat: pd.DataFrame, max_words: int = 6) -> pd.DataFrame:
    """Короткие (≤ max_words слов) русские фразы BANKING77 поисковых и справочных интентов."""
    intent2f = {
        i: f
        for f, s in zip(cat["feature_id"], cat["b77_intents"], strict=True)
        for i in s.split(";")
        if i
    }
    keep = b77ru[b77ru["intent"].isin(SEARCH_INTENTS + FAQ_INTENTS)].copy()
    keep = keep[keep["text"].str.split().str.len() <= max_words]
    return pd.DataFrame(
        {
            "text": keep["text"].to_numpy(),
            "source": "banking77_ru",
            "stratum": keep["intent"]
            .map(lambda i: "search" if i in SEARCH_INTENTS else "faq")
            .to_numpy(),
            "targets": keep["intent"].map(lambda i: [intent2f[i]]).to_numpy(),
            "intent": keep["intent"].to_numpy(),
        }
    )


def ssq_queries(ssq: pd.DataFrame) -> pd.DataFrame:
    """Финансовые запросы ssq_ru с целями по правилам (пустые цели — вне каталога)."""
    t = ssq["text"].map(ssq_targets)
    return pd.DataFrame(
        {
            "text": ssq["text"].to_numpy(),
            "source": "ssq_ru",
            "stratum": ssq["category"].to_numpy(),
            "targets": t.to_numpy(),
        }
    )


def build_qrels(queries: pd.DataFrame, cat: pd.DataFrame) -> pd.DataFrame:
    """Плотная разметка всех пар «запрос × фича»: 2 — целевая, 1 — тот же раздел, 0 — иначе."""
    section = dict(zip(cat["feature_id"], cat["section"], strict=True))
    rows = []
    for qid, targets in zip(queries["query_id"], queries["targets"], strict=True):
        tsec = {section[t] for t in targets}
        for fid in cat["feature_id"]:
            rel = 2 if fid in targets else 1 if section[fid] in tsec else 0
            rows.append((qid, fid, rel))
    return pd.DataFrame(rows, columns=["query_id", "doc_id", "rel"])
