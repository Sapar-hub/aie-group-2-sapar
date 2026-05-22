# Тесты проекта

## Файлы тестов

| Файл | Описание | Тестов | Статус |
|------|----------|--------|--------|
| `test_metrics.py` | Модульные тесты метрик: IoU, YOLO conversion, precision/recall/F1 | 10 | ✅ Все проходят |
| `test_models.py` | Тесты моделей: CV baseline, YOLO (mock), Hybrid refiner | 6 | ✅ 5 проходят, 1 пропущен |
| `test_pipeline.py` | End-to-end sanity checks | — | ❌ Не реализован |

## Запуск

```bash
cd project
pytest tests -v
```

**Текущий результат:** 15 passed, 1 skipped (0.13s)

Пропущенный тест: `TestRCNNModel::test_model_init` — требует установки torch/torchvision (тестируется вручную на Colab с GPU).
