# MLSD Course

Учебный репозиторий проекта по курсу **Machine Learning System Design**.

Публичный репозиторий: [github.com/badmax333/MLSD_course](https://github.com/badmax333/MLSD_course)

## О проекте

**ML-персонализация удержания клиентов ММБ** — ML- и LLM-подходы для прогнозирования причин оттока клиентов ММБ (ИП и ООО с годовой выручкой до 500 млн руб.), ранжирования инструментов удержания (топ-3 сценария) и генерации персональных сейлз-аргументов для голосового и цифрового каналов.

Дизайн-документ: [docs/ml_system_design_doc.md](docs/ml_system_design_doc.md)

Здесь будет код, эксперименты и документация ML-системы, которую мы проектируем в рамках курса. Сейчас репозиторий подготовлен к разработке: зафиксированы зависимости, линтер, форматер и проверки перед коммитом.

## Стек

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) — окружение и зависимости
- [Ruff](https://docs.astral.sh/ruff/) — линтер и форматер
- [pre-commit](https://pre-commit.com/) — автоматические проверки перед коммитом

## Быстрый старт

Нужны [Git](https://git-scm.com/) и [uv](https://docs.astral.sh/uv/getting-started/installation/).

```bash
git clone https://github.com/badmax333/MLSD_course.git
cd MLSD_course
uv sync --group dev
uv run pre-commit install
```

`pre-commit install` ставит git-хук: перед каждым коммитом прогоняются линтер, форматер и базовые проверки файлов.

## Качество кода

Ruff и линтит, и форматирует. Правила заданы в `pyproject.toml`: стиль, неиспользуемый код, сортировка импортов, pyupgrade, bugbear и правила самого Ruff. Длина строки — 100 символов.

Линтер:

```bash
uv run ruff check .
```

Форматер:

```bash
uv run ruff format .
```

Проверить форматирование без правок:

```bash
uv run ruff format --check .
```

Все хуки pre-commit по всему репозиторию:

```bash
uv run pre-commit run --all-files
```

## Структура

```text
.
├── .pre-commit-config.yaml   # хуки pre-commit
├── pyproject.toml            # проект и настройки Ruff
├── uv.lock                   # зафиксированные зависимости
└── src/mlsd_course/          # исходный код
```

## Автор

[badmax333](https://github.com/badmax333)
