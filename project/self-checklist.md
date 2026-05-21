# Самопроверка проекта (self-checklist)

| # | Критерий | Да/Нет | Где смотреть / комментарий |
|---|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `README.md` и работает | ✅ | `src/api/main.py`, `README.md` раздел 4.4 |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/api/main.py` — YOLO или CV baseline |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/exp01_eda_baseline.ipynb`, `exp04_yolo_experiments.ipynb` |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ✅ | CV baseline (exp03) + YOLO (exp04) + RCNN (exp05) + Hybrid (exp06) + Comparison (exp07) |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/evaluation/`, `src/hybrid/`, `src/api/` |
| 6 | Есть Dockerfile **или** понятный сценарий развёртывания без Docker | ✅ | `README.md` раздел 4 — pip install + python -m src.api.main |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов | ✅ | `configs/.env.example`, `.gitignore` исключает `.env` |
| 8 | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`) | ✅ | `src/api/main.py` — logging + `/health` endpoint |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам | ✅ | `report.md` раздел 5 — таблица сравнения + вывод |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md` раздел 4 (команды), `report.md` раздел 9 (сценарий) |
