"""Текстовая статистика запросов: длины, дубли, шум, близость интентов."""

from __future__ import annotations

import re
from collections import Counter

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)
CYR_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LAT_RE = re.compile(r"[a-z]", re.IGNORECASE)


def normalize(text: str) -> str:
    """Нижний регистр, только буквы и цифры, одиночные пробелы."""
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())


def duplicates_report(df: pd.DataFrame, text_col: str = "text", label_col: str = "intent") -> dict:
    """Точные дубли после нормализации и дубли с разными метками."""
    norm = df[text_col].map(normalize)
    groups = df.assign(_norm=norm).groupby("_norm")[label_col].agg(["size", "nunique"])
    dup = groups[groups["size"] > 1]
    conflict = groups[groups["nunique"] > 1]
    return {
        "n_rows": len(df),
        "n_unique_norm": len(groups),
        "n_dup_groups": len(dup),
        "n_rows_in_dup_groups": int(dup["size"].sum()),
        "n_conflict_groups": len(conflict),
        "n_rows_in_conflict_groups": int(conflict["size"].sum()),
        "conflict_examples": conflict.sort_values("size", ascending=False).head(10).index.tolist(),
    }


def _deletes1(word: str) -> set[str]:
    return {word[:i] + word[i + 1 :] for i in range(len(word))}


def typo_candidates(
    texts: pd.Series, rare_max: int = 1, frequent_min: int = 10, min_len: int = 4
) -> pd.DataFrame:
    """Редкие токены на расстоянии одной правки (вставка/удаление/замена) от частого токена.

    Это эвристика: она находит и настоящие опечатки, и редкие легальные формы слова.
    """
    counts = Counter(t for text in texts for t in tokens(text))
    frequent = [w for w, c in counts.items() if c >= frequent_min and len(w) >= min_len - 1]
    index: dict[str, set[str]] = {}
    for w in frequent:
        for key in {w} | _deletes1(w):
            index.setdefault(key, set()).add(w)
    rows = []
    for w, c in counts.items():
        if c > rare_max or len(w) < min_len or w.isdigit():
            continue
        cands = set()
        for key in {w} | _deletes1(w):
            cands |= index.get(key, set())
        cands = {x for x in cands if x != w and abs(len(x) - len(w)) <= 1}
        if cands:
            best = max(cands, key=counts.__getitem__)
            rows.append({"rare": w, "count": c, "likely": best, "likely_count": counts[best]})
    return pd.DataFrame(rows).sort_values("likely_count", ascending=False, ignore_index=True)


def script_mix(text: str) -> str:
    """Тип алфавита в строке: cyr, lat, mixed, none."""
    has_c, has_l = bool(CYR_RE.search(text)), bool(LAT_RE.search(text))
    if has_c and has_l:
        return "mixed"
    return "cyr" if has_c else "lat" if has_l else "none"


def intent_similarity(
    df: pd.DataFrame, text_col: str = "text", label_col: str = "intent"
) -> tuple[pd.DataFrame, list[str]]:
    """Косинусная близость TF-IDF-центроидов интентов (символьные n-граммы 2–5)."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    x = vec.fit_transform(df[text_col].str.lower())
    labels = sorted(df[label_col].unique())
    lab = df[label_col].to_numpy()
    cent = np.vstack([np.asarray(x[lab == name].mean(axis=0)) for name in labels])
    sim = cosine_similarity(cent)
    return pd.DataFrame(sim, index=labels, columns=labels), labels


def top_pairs(sim: pd.DataFrame, k: int = 15) -> pd.DataFrame:
    """Самые близкие пары разных интентов."""
    names = sim.index.to_list()
    vals = sim.to_numpy()
    iu = np.triu_indices(len(names), k=1)
    pairs = pd.DataFrame(
        {
            "intent_a": [names[i] for i in iu[0]],
            "intent_b": [names[j] for j in iu[1]],
            "cos": vals[iu],
        }
    )
    return pairs.sort_values("cos", ascending=False, ignore_index=True).head(k)


def wilson_halfwidth(p: float, n: int, z: float = 1.96) -> float:
    """Полуширина 95% доверительного интервала Уилсона для доли p на n наблюдениях."""
    denom = 1 + z**2 / n
    return float(z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom)
