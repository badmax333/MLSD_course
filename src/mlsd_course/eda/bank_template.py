"""Шаблон EDA для выгрузки банка в формате BANK_SERP_COLUMNS (см. loaders.py).

Одна строка — одна позиция выдачи. Запускается без правок на реальной выгрузке,
а до её появления — на синтетическом каталоге (ноутбук 02, раздел 5).
"""

from __future__ import annotations

import pandas as pd

from mlsd_course.eda.text_stats import normalize


def bank_eda(serp: pd.DataFrame) -> dict:
    """Сводка по выгрузке: объём, баланс интентов, длина выдачи, доля релевантных, качество."""
    q = serp.drop_duplicates(["run_id", "platform", "app_version", "query"])
    per_intent = q.groupby("intent")["query"].nunique()
    top3 = serp[serp["position"] <= 3]
    return {
        "прогонов": serp["run_id"].nunique(),
        "платформ / версий": (serp["platform"].nunique(), serp["app_version"].nunique()),
        "запросов уникальных": serp["query"].nunique(),
        "интентов": serp["intent"].nunique(),
        "запросов на интент min/median/max": (
            int(per_intent.min()),
            float(per_intent.median()),
            int(per_intent.max()),
        ),
        "интентов с < 50 запросами": int((per_intent < 50).sum()),
        "средняя длина выдачи": round(
            float(serp.groupby(["run_id", "query"])["position"].max().mean()), 2
        ),
        "доля релевантных в топ-3": round(float((top3["relevance"] > 0).mean()), 3),
        "доля целевых (rel = 2) в топ-3": round(float((top3["relevance"] == 2).mean()), 3),
        "доля вердиктов not_ok": round(float((q["verdict"] == "not_ok").mean()), 3),
        "пустых объяснений": int(q["explanation"].fillna("").str.strip().eq("").sum()),
        "одинаковый текст с разными интентами": int(
            (serp.groupby(serp["query"].map(normalize))["intent"].nunique() > 1).sum()
        ),
    }
