# План: Детекция штампов ГОСТ — гибрид YOLO + CV

## Цель

Построить и сравнить 4 подхода для детекции штампов ГОСТ на строительных чертежах, затем объединить лучший CNN с CV-уточнением в гибрид. Развернуть как FastAPI сервис.

---

## Подходы

| # | Подход | Тип | Обучающие данные |
|---|---|---|---|
| 1 | **CV Baseline** (контуры, template matching, цветовая пороговая обработка) | Классический | Нет |
| 2 | **YOLOv8n** | Одностадийный CNN | 49 реальных + 500 синтетических |
| 3 | **Faster R-CNN** | Двухстадийный CNN | 49 реальных + 500 синтетических |
| 4 | **DETR** | Трансформерный детектор | — (stretch goal, не реализован) |
| 5 | **Hybrid** (YOLO + CV уточнение) | Комбинированный | Лучшая конфигурация CNN |

---

## Протокол

- **Разделение:** 500 synthetic train / 49 real val (single split, без 5-fold CV в финальной версии)
- **Синтетическая генерация:** Crop stamp regions → paste на не-stamp участки с random transforms + рисование по ГОСТ
- **Аугментации:** Horizontal flip, rotation ±15°, scale jitter, brightness/contrast
- **Основная метрика:** IoU (Intersection over Union)
- **Вторичные метрики:** Precision, Recall, F1, Detection rate, inference time

---

## Статус реализации

### Фаза 1: Пайплайн данных
- [x] `src/data/loader.py` — загрузка изображений + YOLO-разметок
- [x] `src/data/augment.py` — аугментации
- [x] `src/data/synthetic.py` — генерация синтетики (GOST + copy-paste)
- [x] `src/data/image_quality.py` — PPI detection + Laplacian variance + donor selection
- [x] `scripts/generate_synthetic.py` — CLI, авто-выбор доноров
- [x] `notebooks/exp01_eda_baseline.ipynb` — EDA
- [x] `notebooks/exp02_synthetic_data.ipynb` — синтетические данные

### Фаза 2: CV Baseline
- [x] `src/models/cv_baseline.py` — contour detection, template matching
- [x] `notebooks/exp03_cv_baseline.ipynb` — результаты, param sweep

### Фаза 3: YOLO
- [x] `src/models/yolo_model.py` — train/inference wrapper
- [x] `notebooks/exp04_yolo_experiments.ipynb` — обучение, метрики

### Фаза 4: Faster R-CNN
- [x] `src/models/rcnn_model.py` — train/inference wrapper
- [x] `notebooks/exp05_faster_rcnn.ipynb` — обучение, результаты

### Фаза 5: DETR (stretch goal)
- [ ] `src/models/detr_model.py` — не реализован
- [ ] `notebooks/exp06_detr.ipynb` — не реализован

### Фаза 6: Hybrid
- [x] `src/hybrid/refiner.py` — CV пост-обработка CNN-кандидатов
- [x] `notebooks/exp06_hybrid.ipynb` — код есть, выходные ячейки отсутствуют

### Фаза 7: Сравнение
- [x] `src/evaluation/metrics.py` — метрики
- [x] `notebooks/exp07_comparison.ipynb` — код есть, выходные ячейки отсутствуют

### Фаза 8: API сервис
- [x] `src/api/main.py` — FastAPI приложение (/health, /predict)
- [x] `src/api/schemas.py` — Pydantic схемы
- [ ] `src/api/router.py` — заглушка, требует доработки или удаления

### Фаза 9: Инфраструктура проекта
- [x] `configs/config.yaml` — все параметры
- [x] `configs/.env.example` — шаблон секретов
- [x] `requirements.txt` — зависимости (требуется исправление cv2 → opencv-python)
- [x] `pyproject.toml` — метаданные + entry points
- [x] `tests/test_metrics.py` — 10 тестов метрик
- [ ] `tests/test_pipeline.py` — end-to-end проверки

### Фаза 10: Документация
- [x] `README.md` — основной README (обновлён)
- [x] `report.md` — отчёт (заполнен частично)
- [x] `self-checklist.md` — самопроверка
- [x] README в `data/`, `configs/`, `notebooks/`, `src/`, `artifacts/`, `tests/`
