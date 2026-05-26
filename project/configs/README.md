# Конфигурационные файлы

## Назначение

В этой папке хранятся настройки проекта, вынесенные из кода для удобства воспроизведения экспериментов.

## Файлы

### `config.yaml` — основной конфиг

Секции:
- `paths` — пути к данным, артефактам, синтетике
- `cv_baseline` — параметры CV-детектора (aspect ratio, adaptive threshold)
- `yolo` — параметры YOLO (model_name, epochs, imgsz, batch, device)
- `rcnn` — параметры Faster R-CNN (epochs, batch_size, lr, imgsz, device, patience, target_ppi, synth_dpi, max_size)
- `hybrid` — параметры гибридного уточнения (cnn_weight, IoU threshold, AR tolerance)
- `synthetic` — параметры генерации синтетики (donor_count, num_gost, num_copypaste)
- `evaluation` — пороги IoU + путь к весам YOLO для оценки
- `model.classes` — список классов (только `stamp`)

### `.env.example` — шаблон переменных окружения

```
MODEL_PATH=artifacts/models/best.pt
API_HOST=0.0.0.0
API_PORT=8000
LOG_LEVEL=INFO
```

Реальные `.env` файлы не коммитятся в репозиторий (игнорируются `.gitignore`).
