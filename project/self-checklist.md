# Самопроверка проекта (self-checklist)

| # | Критерий | Статус | Где смотреть / комментарий |
|---|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `README.md` и работает | ✅ | `src/api/main.py` — FastAPI приложение, запуск `python -m src.api.main`. Модель: `artifacts/models/best.pt` (symlink → `yolo/v4/weights/best.pt`). |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/api/main.py` — загружает YOLO (`artifacts/models/best.pt`), при падении — CV baseline. Обе модели реальные. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/exp01_eda_baseline.ipynb` (EDA) + `exp04_yolo_experiments.ipynb` (YOLO). |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ✅ | Все метрики собраны: CV baseline (F1 0.727), YOLO v3 (F1 0.871), YOLO v4 (F1 0.754), Hybrid v4 (F1 0.712), RCNN v4 (F1 0.627). Таблица в `README.md` раздел 5 и `report.md` раздел 5. |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/evaluation/`, `src/hybrid/`, `src/api/` — модульная структура. |
| 6 | Есть Dockerfile **или** понятный сценарий развёртывания без Docker | ✅ | `Dockerfile` + `docker-compose.yml` — многоступенчатая сборка, healthcheck, volume для весов. |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов | ✅ | `configs/.env.example`, `.env` в `.gitignore`. |
| 8 | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`) | ✅ | `src/api/main.py` — structured logging + `/health` endpoint. |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам | ✅ | Все модели сравниваются на 35 real (v4). **Финальный выбор:** YOLOv8n v4 (F1 0.754) — лучший F1 среди воспроизводимых моделей. v3 невоспроизводим за пределами Colab. |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md` раздел 4 (команды запуска), `report.md` раздел 9 (сценарий защиты). |
