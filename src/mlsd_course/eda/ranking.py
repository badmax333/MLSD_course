"""Метрики выдачи: R3, DCG, NDCG@k.

Метрики считает детерминированный код по оценкам релевантности позиций, а не LLM.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd


def r_at_k(rels: Sequence[float], k: int = 3) -> float:
    """R@k (R3 при k=3): доля релевантных среди первых k позиций (rel > 0)."""
    top = list(rels[:k])
    if not top:
        return 0.0
    return sum(1 for r in top if r > 0) / k


def dcg_at_k(rels: Sequence[float], k: int = 3) -> float:
    """DCG@k с экспоненциальным усилением: sum (2^rel - 1) / log2(pos + 1)."""
    arr = np.asarray(rels[:k], dtype=float)
    if arr.size == 0:
        return 0.0
    disc = np.log2(np.arange(2, arr.size + 2))
    return float(((2**arr - 1) / disc).sum())


def ndcg_at_k(rels: Sequence[float], k: int = 3, ideal: Sequence[float] | None = None) -> float:
    """NDCG@k. ideal — все известные оценки для запроса (по умолчанию — те же rels)."""
    ideal_rels = sorted(ideal if ideal is not None else rels, reverse=True)
    idcg = dcg_at_k(ideal_rels, k)
    return dcg_at_k(rels, k) / idcg if idcg > 0 else 0.0


def evaluate_run(run: pd.DataFrame, qrels: pd.DataFrame, k: int = 3) -> pd.DataFrame:
    """R@k и NDCG@k по запросам для выдачи с внешней разметкой пар.

    run: query_id, position, doc_id. qrels: query_id, doc_id, rel (0/1/2).
    Неразмеченная позиция считается нерелевантной (rel = 0), её долю возвращаем в ``labeled``.
    Идеальная выдача для NDCG — все размеченные пары запроса, отсортированные по rel.
    """
    top = run[run["position"] <= k].merge(qrels, on=["query_id", "doc_id"], how="left")
    top = top.sort_values(["query_id", "position"])
    ideal = qrels.groupby("query_id")["rel"].apply(list).to_dict()
    out = []
    for qid, g in top.groupby("query_id", sort=False):
        rels = g["rel"].fillna(0).tolist()
        out.append(
            {
                "query_id": qid,
                f"r{k}": r_at_k(rels, k),
                f"ndcg{k}": ndcg_at_k(rels, k, ideal=ideal.get(qid, [])),
                "labeled": g["rel"].notna().mean(),
            }
        )
    return pd.DataFrame(out)
