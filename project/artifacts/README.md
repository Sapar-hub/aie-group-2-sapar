# Артефакты проекта

## Структура

```
artifacts/
├── models/                       # Symlink'и на актуальные веса
│   ├── best.pt                   → ../yolo/v4/weights/best.pt
│   └── rcnn_best.pth             → ../rcnn/v4/rcnn_best.pth
├── yolo/
│   ├── v4/                       # ← Текущая YOLO-модель (exp04_v4, 50 эпох)
│   │   ├── weights/
│   │   │   ├── best.pt           # 6 MB
│   │   │   └── last.pt
│   │   ├── PR_curve.png
│   │   ├── confusion_matrix.png
│   │   ├── results.csv
│   │   └── ...
│   └── legacy/                   # Исторические версии (невоспроизводимы)
│       ├── v3/                   # exp04 — shortcut "белая сетка = штамп"
│       │   ├── weights/best.pt
│       │   ├── yolo_v3_results.json
│       │   └── ...
│       ├── v1/                   # утечка фона
│       ├── v2/                   # фоны исправлены
│       └── rcnn_v2/              # Faster R-CNN v2
├── rcnn/
│   └── v4/                       # Faster R-CNN v4 (current)
│       └── rcnn_best.pth         # 159 MB
├── figures/                      # Визуализации (только v4 + cross-model)
│   ├── comparison_*.png
│   ├── yolo_v4_best_worst.png
│   ├── rcnn_v4_*.png
│   ├── hybrid_best_worst.png
│   └── eda/
├── metrics/                      # Результаты экспериментов (только v4 + cross-model)
│   ├── comparison_results.json
│   ├── cv_baseline_results.json
│   ├── hybrid_results.json
│   ├── rcnn_v4_results.json
│   ├── yolo_v4_results.json
│   ├── eda_summary.json
│   └── donors.txt
├── pretrained/                   # Предобученные веса (скачиваются при первой загрузке)
└── README.md                     # Этот файл
```

## Принцип переключения модели

Обновить symlink — и API автоматом использует другую версию:

```bash
# На v3 (legacy)
ln -sf ../yolo/legacy/v3/weights/best.pt artifacts/models/best.pt

# На v4 (current)
ln -sf ../yolo/v4/weights/best.pt artifacts/models/best.pt

# Через переменную окружения
MODEL_PATH=/path/to/model.pt python -m src.api.main
```

## Замечания

- v4 — единственная полностью воспроизводимая версия данных
- v1–v3 сохранены как история утечек — см. `data/LEAKAGE.md`
- Метрики legacy-версий лежат в соответствующих поддиректориях `yolo/legacy/`
