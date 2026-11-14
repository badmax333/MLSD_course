"""Согласие LLM-судьи с людьми: матрица ошибок, accuracy, κ Коэна, бинарные свёртки."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ESCI = ["E", "S", "C", "I"]

# Свёртки шкалы ESCI (Exact / Substitute / Complement / Irrelevant).
FOLDS = {
    "E/S/C/I (4 класса)": {"E": "E", "S": "S", "C": "C", "I": "I"},
    "наша шкала 2/1/0 (E→2, S→1, C+I→0)": {"E": 2, "S": 1, "C": 0, "I": 0},
    "E против остальных": {"E": 1, "S": 0, "C": 0, "I": 0},
    "E+S против C+I": {"E": 1, "S": 1, "C": 0, "I": 0},
}


def agreement(human: pd.Series, judge: pd.Series, fold: dict) -> dict:
    """Accuracy, κ Коэна и (для порядковой шкалы 2/1/0) квадратично взвешенная κ."""
    h, j = human.map(fold), judge.map(fold)
    out = {
        "n": len(h),
        "accuracy": float((h == j).mean()),
        "kappa": float(cohen_kappa_score(h, j)),
    }
    if set(fold.values()) == {0, 1, 2}:
        out["kappa_quadratic"] = float(cohen_kappa_score(h, j, weights="quadratic"))
    return out


def confusion(human: pd.Series, judge: pd.Series, labels: list) -> pd.DataFrame:
    """Строки — человек, столбцы — судья."""
    m = confusion_matrix(human, judge, labels=labels)
    return pd.DataFrame(
        m, index=[f"человек {x}" for x in labels], columns=[f"судья {x}" for x in labels]
    )


def row_normalized(human: pd.Series, judge: pd.Series, labels: list) -> np.ndarray:
    """P(судья = j | человек = i): эмпирическая модель шума судьи для симуляций."""
    m = confusion_matrix(human, judge, labels=labels).astype(float)
    return m / m.sum(axis=1, keepdims=True)
