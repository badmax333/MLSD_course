"""Загрузка прокси-датасетов и данных банка в ожидаемом формате."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"

BANKING77_URLS = {
    "train": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv",
    "test": "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv",
}
MASSIVE_URL = (
    "https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz"
)

# Ожидаемый формат выгрузки банка (допущение до ответа на В1): одна строка = одна позиция выдачи.
BANK_SERP_COLUMNS = [
    "run_id",  # идентификатор прогона
    "platform",  # android / ios / web
    "app_version",  # версия приложения
    "query",  # текст запроса
    "intent",  # интент из разметки
    "position",  # позиция фичи в выдаче, с 1
    "feature",  # название фичи
    "relevance",  # оценка разметчика для позиции (0/1 или шкала)
    "verdict",  # вердикт по выдаче целиком (ok / not_ok)
    "explanation",  # объяснение разметчика
]


def load_banking77(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """BANKING77: train + test с колонками text, intent, split."""
    parts = []
    for split, url in BANKING77_URLS.items():
        path = raw_dir / f"banking77_{split}.csv"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            pd.read_csv(url).to_csv(path, index=False)
        df = pd.read_csv(path).rename(columns={"category": "intent"})
        df["split"] = split
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def load_massive_ru(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """MASSIVE 1.1, локаль ru-RU. Файл извлекается из архива (см. data/README.md)."""
    path = raw_dir / "massive_ru-RU.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} не найден. Инструкция по скачиванию: data/README.md")
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    df = pd.DataFrame(rows).rename(columns={"utt": "text", "partition": "split"})
    return df


def load_bank_serp(path: Path) -> pd.DataFrame:
    """Выгрузка выдачи банка (CSV или Parquet) в ожидаемом формате BANK_SERP_COLUMNS."""
    df = pd.read_parquet(path) if path.suffix == ".parquet" else pd.read_csv(path)
    missing = set(BANK_SERP_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"В выгрузке нет колонок: {sorted(missing)}")
    return df[BANK_SERP_COLUMNS]
