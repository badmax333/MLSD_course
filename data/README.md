# Данные

Данных банка (запросы, выдача, разметка) в репозитории нет и не будет: это внутренние данные Альфа-Банка. Папка `data/raw/` исключена из git через `.gitignore`.

Пока данных банка нет, EDA выполнен на публичных **прокси-датасетах** и на **синтетическом каталоге фич** (допущение, см. ниже). Прокси моделируют отдельные свойства будущих данных, но не заменяют их. Выбор прокси — по итогам аудита: подробный паспорт с тем, что каждый датасет моделирует и чего не моделирует, — в разделе 1 ноутбука [`02_eda_search_proxies.ipynb`](../notebooks/02_eda_search_proxies.ipynb) и в [отчёте EDA](../reports/eda/README.md).

| Датасет | Что внутри | Роль в проекте | Лицензия | Файлы в `data/raw/` |
| --- | --- | --- | --- | --- |
| [WANDS](https://github.com/wayfair/WANDS) (Wayfair) | 480 поисковых запросов, 42 994 товара, 233 448 пар «запрос–товар» с метками Exact / Partial / Irrelevant | **Основной**: шкала 0/1/2, бейзлайн, R3 и NDCG@3, просадки на реальной выдаче | MIT | `wands/query.csv`, `wands/product.csv`, `wands/label.csv` |
| [ESCI LLM-judged top-10](https://huggingface.co/datasets/johannhartmann/esci-llm-judged-top10) | 1 000 запросов Amazon ESCI, 21 292 пары top-10, 1 500 калибровочных пар «LLM-судья против человека» | **Судья**: согласие с людьми, модель шума | Apache-2.0 | `esci_judge/{calibration,judgments,queries}.parquet` |
| [BANKING77](https://github.com/PolyAI-LDN/task-specific-datasets) ([HF](https://huggingface.co/datasets/PolyAI/banking77)) | 13 083 англоязычных обращения клиентов банка, 77 интентов | Словарь банковских интентов: поиск / справка / тикет | CC BY 4.0 | `banking77_train.csv`, `banking77_test.csv` |
| [banking77_ru](https://huggingface.co/datasets/DeepPavlov/banking77_ru) (DeepPavlov) | Русский перевод BANKING77: 10 003 + 3 080 фраз, те же 77 интентов | Русские формулировки интентов, кривая обучения, запросы к каталогу | в карточке не указана | `banking77_ru/{train,test,intents}.parquet` |
| [synthetic-search-queries-ru](https://huggingface.co/datasets/EmbeddingStudio/synthetic-search-queries-ru) | Русские поисковые запросы, сгенерированные GPT-4; берём 9 финансовых категорий — 923 запроса (757 train + 166 test) | Русские финансовые запросы к каталогу (стресс-набор) | Apache-2.0 | `ssq_ru/{train,test}.parquet` |
| [MASSIVE 1.1](https://github.com/alexa/massive), локаль ru-RU | 16 521 русская команда голосовому ассистенту, 60 интентов, по 3 оценки проверяющих | **Контроль вне домена**, разговорный русский | CC BY 4.0 | `massive_ru-RU.jsonl` |

## Как скачать

Загрузчики в [`src/mlsd_course/eda/loaders.py`](../src/mlsd_course/eda/loaders.py) сами скачивают недостающие файлы при первом вызове, кроме MASSIVE. Вручную то же самое:

```bash
mkdir -p data/raw && cd data/raw

# BANKING77
curl -sL -o banking77_train.csv https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/train.csv
curl -sL -o banking77_test.csv  https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/master/banking_data/test.csv

# WANDS: три TSV-файла (несмотря на расширение .csv), product.csv ~90 МБ
mkdir -p wands
for f in query product label; do
  curl -sL -o wands/$f.csv https://raw.githubusercontent.com/wayfair/WANDS/main/dataset/$f.csv
done

# ESCI LLM-judged top-10 (~2,8 МБ)
mkdir -p esci_judge
for f in calibration judgments queries; do
  curl -sL -o esci_judge/$f.parquet https://huggingface.co/datasets/johannhartmann/esci-llm-judged-top10/resolve/main/data/$f.parquet
done

# banking77_ru
mkdir -p banking77_ru
B=https://huggingface.co/datasets/DeepPavlov/banking77_ru/resolve/main
curl -sL -o banking77_ru/train.parquet   $B/data/train-00000-of-00001.parquet
curl -sL -o banking77_ru/test.parquet    $B/data/test-00000-of-00001.parquet
curl -sL -o banking77_ru/intents.parquet $B/intents/intents-00000-of-00001.parquet

# synthetic-search-queries-ru
mkdir -p ssq_ru
S=https://huggingface.co/datasets/EmbeddingStudio/synthetic-search-queries-ru/resolve/main/data
curl -sL -o ssq_ru/train.parquet $S/train-00000-of-00001.parquet
curl -sL -o ssq_ru/test.parquet  $S/test-00000-of-00001.parquet

# MASSIVE 1.1: архив ~40 МБ, из него нужна только локаль ru-RU
curl -sL -o massive.tar.gz https://amazon-massive-nlu-dataset.s3.amazonaws.com/amazon-massive-dataset-1.1.tar.gz
tar -xzf massive.tar.gz 1.1/data/ru-RU.jsonl && mv 1.1/data/ru-RU.jsonl massive_ru-RU.jsonl && rm -rf 1.1 massive.tar.gz
```

Через `datasets.load_dataset("AmazonScience/massive")` MASSIVE не загружается: на Hugging Face он оформлен скриптом `massive.py`, а новые версии `datasets` скрипты не поддерживают. Поэтому берём исходный архив. Всего данных — около 110 МБ, всё считается на CPU.

## Синтетический каталог фич (допущение)

[`synthetic/alfa_feature_catalog.csv`](synthetic/alfa_feature_catalog.csv) — **не данные банка**, а написанный вручную каталог из 48 карточек: продукты и разделы приложения, сгруппированные в 13 разделов. Он нужен, чтобы до прихода разметки проверить формат «запрос → ожидаемая фича, оценка 0/1/2», бейзлайн и шаблон EDA на русском языке.

| Колонка | Описание |
| --- | --- |
| `feature_id` | Идентификатор карточки, `F01`–`F48` |
| `title` | Название продукта или раздела |
| `section` | Раздел: карточки одного раздела считаются смежными (оценка 1) |
| `description` | Короткое описание. Вместе с названием — индекс бейзлайна |
| `synonyms` | Синонимы через `;`. В индекс не входят, из них строятся шаблонные запросы |
| `b77_intents` | Связанные интенты BANKING77 через `;`. Через них фразы banking77_ru получают целевую фичу |

Запросы к каталогу и метки строит [`catalog.py`](../src/mlsd_course/eda/catalog.py): фразы banking77_ru поисковых и справочных интентов, финансовые запросы ssq_ru, сопоставленные с карточками регулярками, и шаблоны (чистый синоним, опечатка, неверная раскладка, с брендом). Целевая фича — 2, другая фича того же раздела — 1, остальные — 0. Группировка интентов BANKING77 на поиск, справку и тикеты — в [`banking_groups.py`](../src/mlsd_course/eda/banking_groups.py), это тоже ручное допущение.

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

Загрузка: `mlsd_course.eda.loaders.load_bank_serp(path)`, CSV или Parquet. Шаблон EDA для такой выгрузки — `mlsd_course.eda.bank_template.bank_eda`, пример запуска на синтетическом каталоге — раздел 5 ноутбука `notebooks/02_eda_search_proxies.ipynb`.
