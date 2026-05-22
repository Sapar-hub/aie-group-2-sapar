# Исходный код проекта

## Структура модулей

```
src/
├── data/           # Подготовка данных
│   ├── loader.py          # Загрузка изображений и YOLO-разметок
│   ├── augment.py         # Аугментации (flip, rotation, scale, brightness)
│   ├── synthetic.py       # Генерация синтетических данных (GOST + copy-paste)
│   └── image_quality.py   # PPI detection, Laplacian variance, donor selection
├── models/         # Модели детекции
│   ├── cv_baseline.py     # CV baseline (contours, template matching, thresholding)
│   ├── yolo_model.py      # YOLOv8 обёртка (train, predict, export)
│   └── rcnn_model.py      # Faster R-CNN обёртка (build, train, predict)
├── evaluation/     # Метрики
│   └── metrics.py         # IoU, precision, recall, F1, detection rate
├── hybrid/         # Гибридное уточнение
│   ├── refiner.py         # CV пост-обработка CNN-кандидатов
│   └── __init__.py
└── api/            # FastAPI сервис
    ├── main.py            # FastAPI приложение (/health, /predict)
    └── schemas.py         # Pydantic схемы (BoundingBox, PredictionResponse)
```

## Точки входа

- `python -m src.api.main` — запуск FastAPI сервиса (localhost:8000)
- `python -m scripts.generate_synthetic` — генерация синтетического датасета
- `python -m scripts.train_all` — обучение YOLO

## Ключевые классы

- `CVBaselineDetector` — классический CV-детектор (без обучения)
- `YOLOModel` — обучение/инференс YOLO через ultralytics
- `RCNNModel` — обучение/инференс Faster R-CNN (torchvision)
- `HybridRefiner` — уточнение CNN-боксов по aspect ratio и edge density
