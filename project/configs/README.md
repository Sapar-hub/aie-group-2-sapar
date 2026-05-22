# Конфигурационные файлы

## Назначение

В этой папке хранятся настройки проекта, вынесенные из кода для удобства воспроизведения экспериментов.

## Файлы

### `config.yaml` — основной конфиг (50 строк)

Секции:
- `paths` — пути к данным, артефактам, синтетике
- `cv_baseline` — параметры CV-детектора (aspect ratio, adaptive threshold)
- `yolo` — параметры YOLO (model_name, epochs, imgsz, batch, device)
- `rcnn` — параметры Faster R-CNN (epochs, batch_size, lr, imgsz)
- `hybrid` — параметры гибридного уточнения (IoU threshold, AR tolerance)
- `synthetic` — параметры генерации синтетики (donor_count, num_gost, num_copypaste)
- `evaluation` — пороги IoU для расчёта метрик (0.3, 0.5, 0.75)
- `model.classes` — список классов (только `stamp`)

### `.env.example` — шаблон переменных окружения

```
MODEL_PATH=artifacts/models/best.pt
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

Реальные `.env` файлы не коммитятся в репозиторий (игнорируются `.gitignore`).
