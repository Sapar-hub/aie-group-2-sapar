# План: Детекция штампов ГОСТ — гибрид YOLO + CV

## Цель

Построить и сравнить 4 подхода для детекции штампов ГОСТ на строительных чертежах, затем объединить лучший CNN с CV-уточнением в гибрид. Развернуть как FastAPI сервис.

---

## Подходы

| # | Подход | Тип | Обучающие данные |
|---|---|---|---|
| 1 | **CV Baseline** (контуры, adaptive threshold, aspect ratio) | Классический | Нет |
| 2 | **YOLOv8n** | Одностадийный CNN | 500 синтетических (v4) |
| 3 | **Faster R-CNN** | Двухстадийный CNN | 500 синтетических (v4) |
| 4 | **Hybrid** (YOLO + CV уточнение, fused scoring) | Комбинированный | YOLO v4 + CV refiner |
| 5 | **DETR** | Трансформер | — (stretch goal, не реализован) |

---

## Протокол

- **Разделение:** 500 synthetic train / 35 real test (single split, без 5-fold CV)
- **Синтетическая генерация:** crop stamp → paste на не-stamp участки + random transforms + рисование по ГОСТ
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
- [x] `src/models/cv_baseline.py` — contour detection, иерархия, adaptive threshold
- [x] `notebooks/exp03_cv_baseline.ipynb` — результаты, param sweep

### Фаза 3: YOLO
- [x] `src/models/yolo_model.py` — train/inference wrapper
- [x] `notebooks/exp04_yolo_experiments.ipynb` — обучение, метрики

### Фаза 4: Faster R-CNN
- [x] `src/models/rcnn_model.py` — train/inference wrapper
- [x] `notebooks/exp05_faster_rcnn.ipynb` — обучение, результаты

### Фаза 5: DETR (stretch goal)
- [ ] `src/models/detr_model.py` — не реализован

### Фаза 6: Hybrid
- [x] `src/hybrid/refiner.py` — fused scoring (CNN conf + CV score)
- [x] `notebooks/exp06_hybrid.ipynb` — код + выходные ячейки (v3+v4)

### Фаза 7: Сравнение
- [x] `src/evaluation/metrics.py` — метрики
- [x] `src/evaluation/comparison.py` — сводное сравнение
- [x] `src/evaluation/evaluate_yolo.py` — оценка YOLO
- [x] `src/evaluation/evaluate_cv.py` — оценка CV
- [x] `src/evaluation/evaluate_hybrid.py` — оценка гибрида
- [x] `artifacts/metrics/hybrid_results.json` — 5 строк сравнения
- [ ] `notebooks/exp07_comparison.ipynb` — код есть, выходные ячейки отсутствуют

### Фаза 8: API сервис
- [x] `src/api/main.py` — FastAPI (/health, /predict)
- [x] `src/api/schemas.py` — Pydantic схемы
- [ ] `src/api/router.py` — заглушка, требует доработки или удаления

### Фаза 9: Инфраструктура
- [x] `configs/config.yaml` — все параметры
- [x] `configs/.env.example` — шаблон секретов
- [x] `requirements.txt` — зависимости
- [x] `pyproject.toml` — метаданные
- [x] `tests/test_metrics.py` — 10 тестов метрик
- [x] `tests/test_models.py` — 6 тестов моделей
- [ ] `tests/test_pipeline.py` — end-to-end проверки

### Фаза 10: Документация
- [x] `README.md` — основной README (обновлён)
- [x] `report.md` — отчёт (заполнен частично)
- [x] `self-checklist.md` — самопроверка (обновлена)
- [x] README в `data/`, `configs/`, `notebooks/`, `src/`, `artifacts/`, `tests/`
