# HW04 – eda_cli: HTTP-сервис качества датасетов

CLI-приложение и HTTP-сервис (FastAPI) для базового анализа CSV-файлов и оценки их качества.

## Требования

- Python 3.11+
- [uv](https://docs.astral.sh/uv/) установлен в систему

## Инициализация проекта

В корне проекта (eda-cli):

```bash
uv sync
```

## Запуск CLI

### Краткий обзор

```bash
uv run eda-cli head data/example.csv --n 5
```

### Полный EDA-отчёт

```bash
uv run eda-cli report data/example.csv --out-dir reports
```

## Запуск HTTP-сервиса

```bash
uv run uvicorn eda_cli.api:app --reload --port 8000
```

После запуска документация доступна по адресу: [http://localhost:8000/docs](http://localhost:8000/docs)

### Эндпоинты

- `GET /health` — проверка работоспособности сервиса.
- `GET /metrics` — статистика работы сервиса (общее кол-во запросов, среднее время, доля качественных датасетов).
- `POST /quality` — оценка качества по агрегированным признакам (JSON).
- `POST /quality-from-csv` — оценка качества по загруженному CSV-файлу.
- `POST /quality-flags-from-csv` — возвращает полный набор флагов качества (доработки HW03).
- `POST /head` — возвращает первые `n` строк загруженного CSV-файла.

## Тесты

```bash
uv run pytest -q
```