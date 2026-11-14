"""Нормализация и токенизация текста запросов."""

from __future__ import annotations

import re

TOKEN_RE = re.compile(r"[^\W_]+", re.UNICODE)


def normalize(text: str) -> str:
    """Нижний регистр, только буквы и цифры, одиночные пробелы."""
    return " ".join(TOKEN_RE.findall(text.lower()))


def tokens(text: str) -> list[str]:
    return TOKEN_RE.findall(text.lower())
