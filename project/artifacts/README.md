# Артефакты проекта

## Структура

```
artifacts/
├── figures/           # Визуализации и графики
│   ├── eda_ppi_distribution.png      # Распределение PPI
│   ├── eda_stamp_sizes.png           # Размеры штампов в пикселях
│   ├── eda_physical_area.png         # Физическая площадь штампов (мм²)
│   ├── eda_resolution_aspect.png     # Разрешение и aspect ratio изображений
│   ├── eda_donor_selection.png       # Визуализация отбора доноров
│   ├── yolo_best_worst.png           # Лучшие/худшие предсказания YOLO
│   └── rcnn_loss.png                 # График потерь Faster R-CNN
├── metrics/           # Результаты экспериментов
│   ├── eda_summary.json              # EDA-метрики (49 изображений)
│   ├── yolo_results.json             # YOLO: IoU 0.88, Precision 1.0, Recall 1.0
│   └── donors.txt                    # 4 файла-донора
├── models/            # Веса моделей (пусто — фактически в yolo/exp01/)
├── pretrained/        # Предобученные веса
├── visualizations/    # Дополнительные визуализации
└── yolo/              # Результаты YOLO-тренировки
    └── exp01/
        ├── weights/
        │   ├── best.pt     # 6 MB — лучшие веса (50 эпох)
        │   └── last.pt     # 6 MB — веса последней эпохи
        ├── results.csv     # Полный лог тренировки (box loss, cls loss, mAP)
        ├── args.yaml       # Параметры тренировки
        ├── PR_curve.png    # Precision-Recall кривая
        ├── confusion_matrix.png
        └── ...             # Batch-визуализации, labels, predictions
```

## Замечания

- Актуальные веса YOLO: `artifacts/yolo/exp01/weights/best.pt`
- Путь `artifacts/models/best.pt` отсутствует — ожидает копирования/ссылки
- Нет метрик Faster R-CNN, гибрида и финального сравнения — ожидают запуска ноутбуков exp06/exp07
