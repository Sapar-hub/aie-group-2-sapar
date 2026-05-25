# Артефакты проекта

## Структура

```
artifacts/
├── figures/           # Визуализации и графики
│   ├── eda_ppi_distribution.png      # Распределение PPI
│   ├── eda_stamp_sizes.png           # Размеры штампов в пикселях
│   ├── eda_physical_area.png         # Физическая площадь штампов (мм²)
│   ├── eda_resolution_aspect.png     # Разрешение и aspect ratio
│   ├── eda_donor_selection.png       # Визуализация отбора доноров
│   ├── yolo_best_worst.png           # Лучшие/худшие предсказания YOLO
│   ├── rcnn_loss.png                 # График потерь Faster R-CNN
│   └── hybrid_best_worst.png         # Лучшие/худшие предсказания гибрида
├── metrics/           # Результаты экспериментов
│   ├── eda_summary.json              # EDA-метрики (49 изображений)
│   ├── donors.txt                    # 4 файла-донора
│   ├── cv_baseline_results.json      # CV baseline
│   ├── yolo_v1_results.json          # YOLO v1 (утечка фона)
│   ├── yolo_v2_results.json          # YOLO v2 (без утечки)
│   ├── yolo_v3_results.json          # YOLO v3 (чистый baseline)
│   ├── yolo_v4_results.json          # YOLO v4 (harder mix)
│   ├── rcnn_v2_results.json          # Faster R-CNN v2
│   └── hybrid_results.json           # Гибрид v3 + v4
├── models/            # Веса моделей
│   ├── rcnn_v2/rcnn_best.pth         # Faster R-CNN v2 (165 MB)
│   ├── rcnn_v4_best.pth              # Faster R-CNN v4 (165 MB)
│   ├── yolo_v1/weights/best.pt       # YOLO v1
│   └── yolo_v2/weights/best.pt       # YOLO v2
├── pretrained/        # Предобученные веса (скачиваются при первой загрузке)
└── yolo/              # Результаты YOLO-тренировки
    ├── exp04/
    │   └── weights/best.pt   # 6 MB — YOLO v3 (50 эпох, train_v3)
    └── exp04_v4/
        └── weights/best.pt   # 6 MB — YOLO v4 (50 эпох, train_v4)
```

## Замечания

- Актуальные веса YOLO v4: `artifacts/yolo/exp04_v4/weights/best.pt`
- Веса YOLO v3: `artifacts/yolo/exp04/weights/best.pt` (исторический baseline)
- v1–v3 метрики сохранены как история утечек — см. `data/LEAKAGE.md`
- Метрики гибрида: `artifacts/metrics/hybrid_results.json`
