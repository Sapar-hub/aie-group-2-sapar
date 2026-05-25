# Самопроверка проекта (self-checklist)

| # | Критерий | Статус | Где смотреть / комментарий |
|---|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `README.md` и работает | ✅ | `src/api/main.py` — FastAPI приложение, запуск `python -m src.api.main`. Модель: `artifacts/yolo/exp04/weights/best.pt`. |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/api/main.py` — загружает YOLO (`best.pt`), при падении — CV baseline. Обе модели реальные. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/exp01_eda_baseline.ipynb` (EDA, 9 ячеек, ✅ выходные данные) + `notebooks/exp04_yolo_experiments.ipynb` (YOLO, 11 ячеек, ✅ выходные данные). |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ✅ | Все метрики собраны: CV baseline v3 (IoU 0.633), YOLO v3 (IoU 0.700), Faster R-CNN v2 (IoU 0.731). Таблица в `README.md` раздел 5 и `report.md` раздел 5. **Ожидает:** Hybrid (exp06) и единое сравнение (exp07). |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/evaluation/`, `src/hybrid/`, `src/api/` — модульная структура. |
| 6 | Есть Dockerfile **или** понятный сценарий развёртывания без Docker | ❌ | **Dockerfile отсутствует.** Развёртывание только через `pip install -r requirements.txt` + `python -m src.api.main`. |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов | ✅ | `configs/.env.example` (MODEL_PATH, API_HOST, API_PORT, LOG_LEVEL), `.env` в `.gitignore`. |
| 8 | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`) | ✅ | `src/api/main.py` — structured logging (time, level, module, message) + `/health` endpoint с статусом модели. |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам | ⚠️ Частично | Таблица метрик обновлена (CV baseline v3, YOLO v3, RCNN v2). Обоснование выбора отложено до exp07 — прямое сравнение некорректно из-за разных тестовых выборок (35 vs 45 изображений). |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md` раздел 4 (команды запуска), `report.md` раздел 9 (сценарий защиты). |
