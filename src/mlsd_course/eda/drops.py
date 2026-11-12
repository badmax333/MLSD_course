"""Симуляция релизов поиска: нейтральные изменения, поломки топ-3 и шумный судья.

Выдача запроса — список (doc_id, rel) по позициям. «Релиз» = базовое ранжирование со слабым
мультипликативным шумом скоров (нейтральное изменение модели поиска) и, у части запросов,
поломка топ-3 одного из типов DROP_KINDS.
"""

from __future__ import annotations

import zlib

import numpy as np

from mlsd_course.eda.ranking import ndcg_at_k

Serp = list[tuple[object, int]]

DROP_KINDS = {
    "swap_1_3": "позиции 1 и 3 меняются местами",
    "shift_best_4": "лучшая позиция топ-3 уезжает на 4-е место",
    "evict_1": "одна релевантная позиция топ-3 заменяется нерелевантной",
    "evict_all": "все релевантные позиции топ-3 заменяются нерелевантными",
}


def rank(
    doc_ids: np.ndarray,
    rels: np.ndarray,
    scores: np.ndarray,
    sigma: float = 0.0,
    rng: np.random.Generator | None = None,
    k: int = 10,
) -> Serp:
    """Top-k по скорам. sigma > 0 — шум скоров exp(N(0, sigma)): нейтральный релиз."""
    if sigma:
        scores = scores * np.exp(rng.normal(0.0, sigma, len(scores)))
    order = np.lexsort((doc_ids, -scores))[:k]
    return list(zip(doc_ids[order].tolist(), rels[order].tolist(), strict=True))


def apply_drop(serp: Serp, kind: str, pool: list[object]) -> Serp:
    """Поломка топ-3 типа kind. pool — нерелевантные документы запроса (rel = 0) вне выдачи."""
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


def release(
    cands: dict[object, tuple[np.ndarray, np.ndarray, np.ndarray]],
    sigma: float,
    seed: int,
    kind: str | None = None,
    share: float = 0.0,
) -> tuple[dict[object, Serp], set]:
    """Новый релиз: шум скоров у всех запросов и поломка kind у доли share запросов,
    у которых в топ-3 есть релевантная позиция. Возвращает выдачу и сломанные запросы."""
    rng = np.random.default_rng(seed)
    cur = {q: rank(*c, sigma=sigma, rng=rng) for q, c in cands.items()}
    hit: set = set()
    if kind and share:
        eligible = sorted(q for q, s in cur.items() if any(r > 0 for _, r in s[:3]))
        n = min(len(eligible), round(share * len(cands)))
        hit = set(rng.choice(eligible, size=n, replace=False).tolist())
        for q in hit:
            shown = {d for d, _ in cur[q]}
            d, r, _ = cands[q]
            pool = [
                x for x, y in zip(d.tolist(), r.tolist(), strict=True) if y == 0 and x not in shown
            ]
            cur[q] = apply_drop(cur[q], kind, pool)
    return cur, hit


def uniform_confusion(error_rate: float) -> np.ndarray:
    """Судья ошибается с вероятностью error_rate, ошибка — одна из двух других меток поровну."""
    m = np.full((3, 3), error_rate / 2)
    np.fill_diagonal(m, 1 - error_rate)
    return m


def judge_label(query: object, doc: object, rel: int, confusion: np.ndarray, salt: int) -> int:
    """Оценка судьи для пары: confusion[i, j] = P(судья ставит j | человек ставит i).

    Одинаковый salt — одинаковый вердикт для пары (кэш вердиктов между прогонами).
    """
    u = (zlib.crc32(f"{salt}|{query}|{doc}".encode()) & 0xFFFFFFFF) / 2**32
    return int(np.searchsorted(np.cumsum(confusion[rel]), u, side="right").clip(0, 2))


def delta_ndcg(
    base: dict[object, Serp],
    cur: dict[object, Serp],
    ideal: dict[object, list[int]],
    confusion: np.ndarray | None = None,
    cache: bool = True,
) -> dict[object, float]:
    """ΔNDCG@3 «релиз − база» по запросам. confusion=None — по разметке людей, иначе по судье.

    cache=False: судья заново оценивает каждую пару в каждом прогоне (независимый шум).
    """
    out = {}
    for q in base:
        rb = [r for _, r in base[q][:3]]
        rc = [r for _, r in cur[q][:3]]
        if confusion is not None:
            sb, sc = (0, 0) if cache else (1, 2)
            rb = [judge_label(q, d, r, confusion, sb) for d, r in base[q][:3]]
            rc = [judge_label(q, d, r, confusion, sc) for d, r in cur[q][:3]]
        out[q] = ndcg_at_k(rc, 3, ideal[q]) - ndcg_at_k(rb, 3, ideal[q])
    return out
