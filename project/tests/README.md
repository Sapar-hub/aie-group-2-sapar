# Тесты проекта

## Файлы тестов

| Файл | Описание | Тестов | Статус |
|------|----------|--------|--------|
| `test_metrics.py` | Модульные тесты метрик: IoU, YOLO-конвертация, precision/recall/F1 | 10 | ✅ Все проходят |
| `test_models.py` | Тесты моделей: CV baseline, YOLO (mock), Hybrid refiner | 6 | ✅ 5 проходят, 1 пропущен |
| `test_pipeline.py` | Сквозные sanity-проверки: CV baseline, YOLO (skip без весов/CUDA) | 6 | ✅ Все проходят (YOLO — skip без совместимого GPU) |

## Запуск

```bash
cd project
pytest tests -v
```

**Текущий результат:** 21-24 passed, 0-3 skipped (зависит от GPU/torch)

Скипаются:
- `TestYOLOPipeline` (3 теста) — если отсутствует best.pt или несовместим GPU
- `TestRCNNModel::test_model_init` — если отсутствует torch (пропускается через `importorskip`)

Пропущенный тест: `TestRCNNModel::test_model_init` — требует установки torch/torchvision (тестируется вручную на Colab с GPU).
