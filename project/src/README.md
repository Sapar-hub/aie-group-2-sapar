# Исходный код проекта

## Структура модулей

```
src/
├── config.py        # Загрузчик конфига (config.yaml), пути, доноры
├── data/            # Подготовка данных
│   ├── loader.py          # Загрузка изображений и YOLO-разметок
│   ├── augment.py         # Аугментации (flip, rotation, scale, brightness)
│   ├── synthetic.py       # Генерация синтетических данных (GOST + copy-paste)
│   └── image_quality.py   # Определение PPI, Laplacian variance, отбор доноров
├── models/          # Модели детекции
│   ├── cv_baseline.py     # CV baseline (контуры, иерархия, adaptive threshold)
│   ├── yolo_model.py      # YOLOv8 обёртка (train, predict, export)
│   └── rcnn_model.py      # Faster R-CNN обёртка (build, train, predict)
├── evaluation/      # Метрики и сравнение
│   ├── metrics.py         # IoU, precision, recall, F1, detection rate
│   ├── evaluate_yolo.py   # Оценка YOLO на тестовой выборке
│   ├── evaluate_cv.py     # Оценка CV baseline
│   ├── evaluate_rcnn.py   # Оценка Faster R-CNN
│   ├── evaluate_hybrid.py # Оценка гибрида (YOLO + CV refine)
│   └── comparison.py      # Сводное сравнение всех моделей
├── hybrid/          # Гибридное уточнение
│   ├── refiner.py         # CV пост-обработка CNN-кандидатов (fused scoring)
│   └── __init__.py
└── api/             # FastAPI сервис
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
- `HybridRefiner` — уточнение CNN-боксов по fused score (CNN conf + CV features)
