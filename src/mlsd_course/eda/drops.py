"""Синтетические просадки на реальной выдаче и детектор просадок с шумным судьёй.

Базовая выдача — реальный top-10 бейзлайна на WANDS с человеческой разметкой.
«Релиз» портит выдачу части запросов, а судья оценивает позиции с заданной ошибкой.
"""

from __future__ import annotations

import zlib

import numpy as np
import pandas as pd

from mlsd_course.eda.ranking import ndcg_at_k

# Типы деградации и их сила. Все действуют только на топ-3, остальное — хвост выдачи.
DROP_KINDS = {
    "swap_1_3": "позиции 1 и 3 меняются местами",
    "shift_best_4": "лучшая по разметке позиция топ-3 уезжает на 4-е место",
    "evict_1": "одна релевантная позиция топ-3 вытесняется нерелевантным товаром",
    "evict_all": "все релевантные позиции топ-3 вытесняются нерелевантными товарами",
}


def apply_drop(
    serp: list[tuple[object, int]], kind: str, pool: list[object]
) -> list[tuple[object, int]]:
    """serp — список (doc_id, rel) по позициям. pool — нерелевантные товары запроса (rel = 0)."""
    s = list(serp)
    top = s[:3]
    if kind == "swap_1_3":
        s[0], s[2] = s[2], s[0]
    elif kind == "shift_best_4":
        best = max(range(3), key=lambda i: (top[i][1], -i))
        s.insert(3, s.pop(best))
    elif kind in {"evict_1", "evict_all"}:
        rel_pos = [i for i in range(3) if top[i][1] > 0]
        if kind == "evict_1":
            rel_pos = rel_pos[:1]
        for i, doc in zip(rel_pos, pool, strict=False):
            s[i] = (doc, 0)
    else:
        raise ValueError(kind)
    return s


def judge_label(
    query_id: object,
    doc_id: object,
    rel: int,
    confusion: np.ndarray,
    run_salt: int,
    cache: bool,
    seed: int,
) -> int:
    """Оценка судьи для пары. confusion[i, j] — P(судья ставит j | человек ставит i).

    cache=True: вердикт пары кэшируется между прогонами (одна и та же пара — одна и та же
    оценка). cache=False: каждый прогон судья оценивает пару заново, с независимым шумом.
    """
    salt = 0 if cache else run_salt
    key = f"{seed}|{salt}|{query_id}|{doc_id}".encode()
    u = (zlib.crc32(key) & 0xFFFFFFFF) / 2**32
    return int(np.searchsorted(np.cumsum(confusion[rel]), u, side="right").clip(0, 2))


def uniform_confusion(error_rate: float) -> np.ndarray:
    """Судья ошибается с вероятностью error_rate, ошибка — одна из двух других меток поровну."""
    m = np.full((3, 3), error_rate / 2)
    np.fill_diagonal(m, 1 - error_rate)
    return m


def benign_change(
    serp: list[tuple[object, int]], rng: np.random.Generator
) -> list[tuple[object, int]]:
    """Нейтральное изменение выдачи: товар из топ-3 меняется на товар из хвоста с той же меткой.

    Метрики по человеческой разметке не меняются, но судья видит новые пары.
    """
    s = list(serp)
    pairs = [(i, j) for i in range(3) for j in range(3, len(s)) if s[i][1] == s[j][1]]
    if pairs:
        i, j = pairs[rng.integers(len(pairs))]
        s[i], s[j] = s[j], s[i]
    return s


def simulate_release(
    serps: dict[object, list[tuple[object, int]]],
    ideal: dict[object, list[int]],
    pools: dict[object, list[object]],
    kind: str,
    share: float = 0.1,
    benign_share: float = 0.3,
    seed: int = 0,
) -> tuple[dict[object, list[tuple[object, int]]], set]:
    """Релиз: у доли share запросов (с релевантным в топ-3) выдача деградирует по типу kind,
    у доли benign_share остальных — меняется нейтрально (см. benign_change).

    Возвращает новую выдачу и множество запросов, где NDCG@3 по разметке реально упал.
    """
    rng = np.random.default_rng(seed)
    eligible = sorted(q for q, s in serps.items() if any(r > 0 for _, r in s[:3]))
    n_hit = max(1, round(share * len(serps)))
    hit = set(rng.choice(eligible, size=min(n_hit, len(eligible)), replace=False).tolist())
    cur = {}
    for q, s in serps.items():
        if q in hit:
            cur[q] = apply_drop(s, kind, pools.get(q, []))
        elif rng.random() < benign_share:
            cur[q] = benign_change(s, rng)
        else:
            cur[q] = s
    real = {
        q
        for q in hit
        if ndcg_at_k([r for _, r in cur[q]], 3, ideal[q])
        < ndcg_at_k([r for _, r in serps[q]], 3, ideal[q]) - 1e-9
    }
    return cur, real


def detect(
    base: dict[object, list[tuple[object, int]]],
    cur: dict[object, list[tuple[object, int]]],
    ideal: dict[object, list[int]],
    confusion: np.ndarray,
    threshold: float,
    cache: bool,
    require_change: bool,
    seed: int = 0,
) -> set:
    """Запросы, где ΔNDCG@3 по оценкам судьи ≤ -threshold.

    require_change=True: просадку подтверждаем, только если топ-3 по doc_id реально изменился.
    """
    flagged = set()
    for q in base:
        if require_change and [d for d, _ in base[q][:3]] == [d for d, _ in cur[q][:3]]:
            continue
        jb = [judge_label(q, d, r, confusion, 1, cache, seed) for d, r in base[q][:3]]
        jc = [judge_label(q, d, r, confusion, 2, cache, seed) for d, r in cur[q][:3]]
        if ndcg_at_k(jc, 3, ideal[q]) - ndcg_at_k(jb, 3, ideal[q]) <= -threshold:
            flagged.add(q)
    return flagged


def pr(flagged: set, real: set) -> dict:
    tp = len(flagged & real)
    return {
        "flagged": len(flagged),
        "tp": tp,
        "precision": tp / len(flagged) if flagged else float("nan"),
        "recall": tp / len(real) if real else float("nan"),
    }


def to_frame(serps: dict[object, list[tuple[object, int]]]) -> pd.DataFrame:
    rows = [
        {"query_id": q, "position": i, "doc_id": d, "rel": r}
        for q, s in serps.items()
        for i, (d, r) in enumerate(s, start=1)
    ]
    return pd.DataFrame(rows)
