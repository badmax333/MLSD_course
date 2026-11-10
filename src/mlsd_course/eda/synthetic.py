"""Синтетическая выдача для иллюстрации R3, NDCG@3 и детектора просадок.

ВНИМАНИЕ: данные полностью синтетические. Каталог фич, запросы и ранжирование
сгенерированы случайно и не отражают реальный поиск банка.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlsd_course.eda.ranking import detect_drops, query_metrics


def make_serp(
    n_intents: int = 20,
    queries_per_intent: int = 50,
    serp_len: int = 8,
    n_features: int = 60,
    seed: int = 0,
) -> pd.DataFrame:
    """Базовая выдача: у каждого интента 1 целевая фича (rel=2) и 2 смежные (rel=1)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_intents):
        target = f"f{3 * i}"
        related = [f"f{3 * i + 1}", f"f{3 * i + 2}"]
        for q in range(queries_per_intent):
            others = [f"f{x}" for x in rng.choice(n_features, size=serp_len, replace=False)]
            others = [f for f in others if f not in {target, *related}]
            # Целевая фича чаще всего первая, иногда вторая или третья.
            pos_target = int(rng.choice([0, 1, 2], p=[0.75, 0.2, 0.05]))
            serp = [*related, *others][: serp_len - 1]
            serp.insert(pos_target, target)
            for pos, feat in enumerate(serp[:serp_len], start=1):
                rel = 2 if feat == target else 1 if feat in related else 0
                rows.append(
                    {
                        "query": f"q{i}_{q}",
                        "intent": f"intent_{i}",
                        "position": pos,
                        "feature": feat,
                        "relevance": rel,
                    }
                )
    return pd.DataFrame(rows)


def inject_release(
    serp: pd.DataFrame, share_queries: float = 0.1, shift: int = 3, seed: int = 1
) -> tuple[pd.DataFrame, set[str]]:
    """Релиз с деградацией: у доли запросов целевая фича съезжает на shift позиций вниз."""
    rng = np.random.default_rng(seed)
    queries = serp["query"].unique()
    hit = set(rng.choice(queries, size=int(len(queries) * share_queries), replace=False))
    out = []
    for query, g in serp.groupby("query", sort=False):
        g = g.sort_values("position").copy()
        if query in hit:
            feats = g["feature"].tolist()
            rels = g["relevance"].tolist()
            t = rels.index(2)
            new_t = min(t + shift, len(feats) - 1)
            feats.insert(new_t, feats.pop(t))
            rels.insert(new_t, rels.pop(t))
            g["feature"], g["relevance"] = feats, rels
        out.append(g)
    return pd.concat(out, ignore_index=True), hit


def noisy_judge(serp: pd.DataFrame, error_rate: float, seed: int = 2) -> pd.DataFrame:
    """Имитация судьи: с вероятностью error_rate оценка позиции заменяется случайной из {0,1,2}."""
    rng = np.random.default_rng(seed)
    out = serp.copy()
    flip = rng.random(len(out)) < error_rate
    out.loc[flip, "relevance"] = rng.integers(0, 3, size=int(flip.sum()))
    return out


def drop_detection_quality(
    base: pd.DataFrame,
    cur: pd.DataFrame,
    true_hit: set[str],
    error_rate: float,
    threshold: float = 0.2,
    seed: int = 2,
) -> dict:
    """Precision/recall детектора просадок, если оба прогона оценивает судья с ошибкой."""
    mb = query_metrics(noisy_judge(base, error_rate, seed=seed))
    mc = query_metrics(noisy_judge(cur, error_rate, seed=seed + 100))
    d = detect_drops(mb, mc, metric="ndcg3", threshold=threshold)
    pred = set(d.loc[d["is_drop"], "query"])
    tp = len(pred & true_hit)
    return {
        "error_rate": error_rate,
        "threshold": threshold,
        "flagged": len(pred),
        "tp": tp,
        "precision": tp / len(pred) if pred else float("nan"),
        "recall": tp / len(true_hit) if true_hit else float("nan"),
    }
