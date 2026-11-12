"""Загрузка прокси-датасетов и данных банка в ожидаемом формате."""

from __future__ import annotations

import json
import urllib.request
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


# --- Прокси для поиска и судьи (ДЗ 2, вторая версия) ---

WANDS_BASE = "https://raw.githubusercontent.com/wayfair/WANDS/main/dataset"
HF = "https://huggingface.co/datasets"
ESCI_JUDGE_BASE = f"{HF}/johannhartmann/esci-llm-judged-top10/resolve/main/data"
BANKING77_RU_BASE = f"{HF}/DeepPavlov/banking77_ru/resolve/main"
SSQ_RU_BASE = f"{HF}/EmbeddingStudio/synthetic-search-queries-ru/resolve/main/data"

# Метки WANDS в шкалу проекта 0/1/2.
WANDS_LABELS = {"Exact": 2, "Partial": 1, "Irrelevant": 0}

# Финансовые категории synthetic-search-queries-ru (отобраны вручную по названию категории).
SSQ_FIN_CATEGORIES = [
    "Banking Services",
    "Digital and Mobile Banking",
    "Credit Services",
    "Investment Services",
    "Insurance Services",
    "Financial Planning and Advisory",
    "Taxation Services",
    "Mortgage and Real Estate Services",
    "Real Estate Investment",
]


def _fetch(url: str, path: Path) -> Path:
    """Скачать файл, если его ещё нет в data/raw/."""
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(url, path)
    return path


def load_wands(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """WANDS (Wayfair): запросы, товары и разметка пар «запрос–товар» (rel: 2/1/0)."""
    out = {}
    for name in ["query", "product", "label"]:
        path = _fetch(f"{WANDS_BASE}/{name}.csv", raw_dir / "wands" / f"{name}.csv")
        out[name] = pd.read_csv(path, sep="\t")
    out["label"]["rel"] = out["label"]["label"].map(WANDS_LABELS)
    return out


def load_esci_judge(raw_dir: Path = RAW_DIR) -> dict[str, pd.DataFrame]:
    """ESCI LLM-judged top-10: пары с оценкой LLM-судьи и калибровка «судья против человека»."""
    out = {}
    for name in ["calibration", "judgments", "queries"]:
        path = _fetch(
            f"{ESCI_JUDGE_BASE}/{name}.parquet", raw_dir / "esci_judge" / f"{name}.parquet"
        )
        out[name] = pd.read_parquet(path)
    return out


def load_banking77_ru(raw_dir: Path = RAW_DIR) -> pd.DataFrame:
    """Русская версия BANKING77 (DeepPavlov): text, intent, split."""
    d = raw_dir / "banking77_ru"
    names = pd.read_parquet(
        _fetch(f"{BANKING77_RU_BASE}/intents/intents-00000-of-00001.parquet", d / "intents.parquet")
    )
    id2name = dict(zip(names["id"], names["name"], strict=True))
    parts = []
    for split in ["train", "test"]:
        url = f"{BANKING77_RU_BASE}/data/{split}-00000-of-00001.parquet"
        df = pd.read_parquet(_fetch(url, d / f"{split}.parquet"))
        df = df.rename(columns={"utterance": "text"})
        df["intent"] = df["label"].map(id2name)
        df["split"] = split
        parts.append(df[["text", "intent", "split"]])
    return pd.concat(parts, ignore_index=True)


def load_ssq_ru(raw_dir: Path = RAW_DIR, fin_only: bool = True) -> pd.DataFrame:
    """synthetic-search-queries-ru: русские поисковые запросы (GPT-4), text, category, split."""
    parts = []
    for split in ["train", "test"]:
        url = f"{SSQ_RU_BASE}/{split}-00000-of-00001.parquet"
        df = pd.read_parquet(_fetch(url, raw_dir / "ssq_ru" / f"{split}.parquet"))
        df = df.rename(columns={"Query": "text"})[["text", "category"]]
        df["split"] = split
        parts.append(df)
    df = pd.concat(parts, ignore_index=True)
    return df[df["category"].isin(SSQ_FIN_CATEGORIES)].reset_index(drop=True) if fin_only else df
