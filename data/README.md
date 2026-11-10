# Данные

Данных банка (запросы, выдача, разметка) в репозитории нет и не будет: это внутренние данные Альфа-Банка. Папка `data/raw/` исключена из git через `.gitignore`.

Пока данных банка нет, EDA выполнен на публичных **прокси-датасетах**. Они моделируют отдельные свойства будущих данных, но не заменяют их.

| Датасет | Что внутри | Лицензия | Файл в `data/raw/` |
| --- | --- | --- | --- |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) ([HF: PolyAI/banking77](https://huggingface.co/datasets/PolyAI/banking77)) | 13 083 англоязычных обращения клиентов банка, 77 интентов, train/test | CC BY 4.0 | `banking77_train.csv`, `banking77_test.csv` |
| [MASSIVE 1.1](https://github.com/alexa/massive), локаль ru-RU | 16 521 русская команда голосовому ассистенту, 60 интентов, 18 сценариев, по 3 оценки проверяющих на пример | CC BY 4.0 | `massive_ru-RU.jsonl` |

## Как скачать

```bash
mkdir -p data/raw && cd data/raw

# BANKING77 (загрузчик скачает и сам, если файлов нет)
curl -sL -o banking77_train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
curl -sL -o banking77_test.csv  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

# MASSIVE 1.1: архив ~40 МБ, из него нужна только локаль ru-RU
curl -sL -o massive.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf massive.tar.gz 1.1/data/ru-RU.jsonl && mv 1.1/data/ru-RU.jsonl massive_ru-RU.jsonl && rm -rf 1.1 massive.tar.gz
```

Через `datasets.load_dataset("AmazonScience/massive")` MASSIVE не загружается: на Hugging Face он оформлен скриптом `massive.py`, а новые версии `datasets` скрипты не поддерживают. Поэтому берём исходный архив.

## Ожидаемый формат данных банка

Допущение до ответа заказчика на вопрос В1. Одна строка — одна позиция выдачи:

| Колонка | Описание |
| --- | --- |
| `run_id` | Идентификатор прогона или партии разметки |
| `platform` | `android`, `ios`, `web` |
| `app_version` | Версия приложения |
| `query` | Текст запроса |
| `intent` | Интент из разметки |
| `position` | Позиция фичи в выдаче, начиная с 1 |
| `feature` | Название фичи |
| `relevance` | Оценка позиции разметчиком: 2 — целевая, 1 — смежная, 0 — нерелевантная |
| `verdict` | Вердикт по выдаче целиком: `ok` или `not_ok` |
| `explanation` | Объяснение разметчика |

Загрузка: `mlsd_course.eda.loaders.load_bank_serp(path)`, CSV или Parquet. Шаблон EDA для такой выгрузки — раздел 7 ноутбука `notebooks/01_eda_proxy.ipynb`.
