# Самопроверка проекта (self-checklist)

| # | Критерий | Статус | Где смотреть / комментарий |
|---|----------|--------|---------------------------|
| 1 | Сервис запускается по инструкциям из `README.md` и работает | ✅ | `src/api/main.py` — FastAPI приложение, запуск `python -m src.api.main`. **Важно:** модель ищется по пути `artifacts/models/best.pt`, но фактически веса лежат в `artifacts/yolo/exp01/weights/best.pt` — требуется копирование. |
| 2 | Endpoint `/predict` использует **реальную модель**, а не заглушку | ✅ | `src/api/main.py` — загружает YOLO (`best.pt`), при падении — CV baseline. Обе модели реальные. |
| 3 | Есть EDA и хотя бы один эксперимент с метриками | ✅ | `notebooks/exp01_eda_baseline.ipynb` (EDA, 9 ячеек, ✅ выходные данные) + `notebooks/exp04_yolo_experiments.ipynb` (YOLO, 11 ячеек, ✅ выходные данные). |
| 4 | Есть baseline и улучшенная модель, есть **сравнение по метрикам** | ⚠️ Частично | CV baseline (exp03, IoU 0.33) и YOLO (exp04, IoU 0.88) — метрики есть. **Отсутствуют:** метрики Faster R-CNN, гибрида и итоговая таблица сравнения. Ноутбуки exp06 и exp07 не выполнены. |
| 5 | Код не свален в один ноутбук: есть структура в `src/` | ✅ | `src/data/`, `src/models/`, `src/evaluation/`, `src/hybrid/`, `src/api/` — модульная структура. |
| 6 | Есть Dockerfile **или** понятный сценарий развёртывания без Docker | ❌ | **Dockerfile отсутствует.** Развёртывание только через `pip install -r requirements.txt` + `python -m src.api.main`. |
| 7 | Есть `.env.example` и **нет** в репозитории реальных секретов | ✅ | `configs/.env.example` (MODEL_PATH, API_HOST, API_PORT, LOG_LEVEL), `.env` в `.gitignore`. |
| 8 | Реализованы логи/наблюдаемость (хотя бы консольные логи + `/health`) | ✅ | `src/api/main.py` — structured logging (time, level, module, message) + `/health` endpoint с статусом модели. |
| 9 | В `report.md` **обоснован выбор финальной модели** по результатам | ❌ | **Таблица метрик в report.md содержит только CV baseline (0.33).** YOLO, RCNN и Hybrid — TODO. Нет обоснования выбора финальной модели. Требуется заполнение после запуска exp06/exp07. |
| 10 | `README.md` и `report.md` позволяют понять сценарий демонстрации | ✅ | `README.md` раздел 4 (команды запуска), `report.md` раздел 9 (сценарий защиты). |
