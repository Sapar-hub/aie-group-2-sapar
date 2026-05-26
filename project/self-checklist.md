# Самопроверка проекта (self-checklist)

| # | Критерий | Статус | Где смотреть / комментарий |
|---|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `README.md` и работает | ✅ | `src/api/main.py` — FastAPI приложение, запуск `python -m src.api.main`. Модель: `artifacts/yolo/exp04_v4/weights/best.pt`. |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/api/main.py` — загружает YOLO (`best.pt`), при падении — CV baseline. Обе модели реальные. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/exp01_eda_baseline.ipynb` (EDA) + `exp04_yolo_experiments.ipynb` (YOLO). |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ✅ | Все метрики собраны: CV baseline (F1 0.719), YOLO v3 (F1 0.857), YOLO v4 (F1 0.678), Hybrid v4 (F1 0.712), RCNN v4 (F1 0.627). Таблица в `README.md` раздел 5 и `report.md` раздел 5. |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/evaluation/`, `src/hybrid/`, `src/api/` — модульная структура. |
| 6 | Есть Dockerfile **или** понятный сценарий развёртывания без Docker | ❌ | **Dockerfile отсутствует.** Развёртывание только через `pip install -r requirements.txt` + `python -m src.api.main`. |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов | ✅ | `configs/.env.example`, `.env` в `.gitignore`. |
| 8 | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`) | ✅ | `src/api/main.py` — structured logging + `/health` endpoint. |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам | ✅ | Все модели сравниваются на 35 real (v4). **Финальный выбор:** CV baseline (F1 0.719) — лучшая воспроизводимая модель. v3 невоспроизводим за пределами Colab. |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md` раздел 4 (команды запуска), `report.md` раздел 9 (сценарий защиты). |
