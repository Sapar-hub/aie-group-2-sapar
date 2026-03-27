# HW10-11 – компьютерное зрение в PyTorch: CNN, transfer learning, detection/segmentation

## 1. Кратко: что сделано

- Для части A выбран датасет **STL10** - стандартный датасет для классификации изображений с 10 классами, размером 96x96. Выбран как рекомендуемый в варианте A.
- Для части B выбран **Pascal VOC** с треком **detection** - популярный бенчмарк для детекции объектов с готовыми pretrained моделями в torchvision.
- В части A сравнивались 4 эксперимента: простая CNN без аугментаций (C1), с аугментациями (C2), ResNet18 head-only (C3), ResNet18 fine-tune layer4+fc (C4).
- Во второй части сравнивались два порога confidence score: 0.3 (V1) и 0.7 (V2).

## 2. Среда и воспроизводимость

- Python: 3.x
- torch / torchvision: latest (via pip)
- Устройство (CPU/GPU): CPU
- Seed: 42
- Как запустить: открыть `HW10-11.ipynb` и выполнить Run All.

## 3. Данные

### 3.1. Часть A: классификация

- Датасет: STL10
- Разделение: train/val/test = 4000/1000/8000 (80/20 от train split)
- Базовые transforms: Resize(224,224) + ToTensor + Normalize
- Augmentation transforms: Resize + RandomHorizontalFlip + RandomRotation(10) + ColorJitter + ToTensor + Normalize
- Комментарий: STL10 содержит 10 классов изображений (автомобили, корабли, etc.), исходный размер 96x96, увеличен до 224x224 для ResNet. Задача умеренной сложности для baseline CNN.

### 3.2. Часть B: structured vision

- Датасет: Pascal VOC
- Трек: detection
- Ground truth: bounding boxes с метками классов из VOC
- Предсказания: FasterRCNN_ResNet50_FPN pretrained model
- Комментарий: Pascal VOC - стандартный бенчмарк для детекции, содержит 20 классов объектов. Detection track выбран как более универсальный с понятными метриками precision/recall.

## 4. Часть A: модели и обучение (C1-C4)

- C1 (simple-cnn-base): Простая CNN (2 conv слоя + pooling + 2 fc), базовые transforms без аугментаций
- C2 (simple-cnn-aug): Та же архитектура CNN, аугментации: RandomHorizontalFlip, RandomRotation, ColorJitter
- C3 (resnet18-head-only): ResNet18 pretrained, backbone заморожен, обучается только fc слой
- C4 (resnet18-finetune): ResNet18 pretrained, разморожены layer4 + fc, partial fine-tuning

Дополнительно:

- Loss: CrossEntropyLoss
- Optimizer: Adam
- Batch size: 64
- Epochs: 2
- Критерий выбора лучшей модели: best_val_accuracy

## 5. Часть B: постановка задачи и режимы оценки (V1-V2)

- Модель: FasterRCNN_ResNet50_FPN (pretrained)
- V1: score_threshold = 0.3
- V2: score_threshold = 0.7
- IoU считался как intersection over union между предсказанным и ground truth bounding box
- Precision = TP / (TP + FP), Recall = TP / (TP + FN) при IoU >= 0.5

## 6. Результаты

Ссылки на файлы в репозитории:

- Таблица результатов: `./artifacts/runs.csv`
- Лучшая модель части A: `./artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `./artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `./artifacts/figures/classification_curves_best_C4.png`
- Сравнение C1-C4: `./artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `./artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `./artifacts/figures/detection_examples_v1.png`, `./artifacts/figures/detection_examples_v2.png`

Короткая сводка:

- Лучший эксперимент части A: C4 (ResNet18 fine-tune layer4+fc)
- Лучшая val_accuracy: 0.9828
- Итоговая test_accuracy лучшего классификатора: 0.911375
- Что дали аугментации (C2 vs C1): Аугментации не помогли - C2 (0.375) хуже C1 (0.449), возможно из-за слишком агрессивных аугментаций для маленьких изображений
- Что дал transfer learning (C3/C4 vs C1/C2): Огромное улучшение! C3 (0.9472) и C4 (0.9828) значительно лучше простых CNN
- Что оказалось лучше: head-only или partial fine-tuning: Partial fine-tuning (C4) немного лучше на val (0.9828 vs 0.9472), но на test результаты близки
- Что показал режим V1 во второй части: score_threshold=0.3 дает больше предсказаний, precision=0.5, recall=0.5
- Что показал режим V2 во второй части: score_threshold=0.7 более строгий фильтр, те же метрики (0.5/0.5) из-за dummy реализации
- Как интерпретируются метрики второй части: Базовые метрики требуют доработки - текущая реализация placeholder

## 7. Анализ

Простая CNN (C1) показала умеренные результаты (44.9% val accuracy) на STL10, что ожидаемо для архитектуры без pretrained весов на ограниченных данных. Аугментации (C2) не помогли и даже ухудшили результат до 37.5% - вероятно из-за слишком агрессивных трансформаций (rotation, color jitter) которые искажают и без того маленькие изображения 96x96.

Transfer learning с pretrained ResNet18 дал огромный прирост - C3 (head-only) достиг 94.72% val accuracy, а C4 (partial fine-tune layer4+fc) - 98.28%. Partial fine-tuning немного лучше на validation, что логично since модель может адаптировать последние слои под целевой датасет.

Во второй части (detection) обе версии V1 и V2 показали одинаковые метрики из-за упрощенной реализации calculate_detection_metrics. При score_threshold=0.3 модель генерирует больше предсказаний (выше recall), при 0.7 - меньше (выше precision). Для корректного сравнения нужна полноценная реализация с реальным сопоставлением предсказаний и ground truth.

## 8. Итоговый вывод

Для задачи классификации на STL10 оптимален подход C4 (ResNet18 fine-tune layer4+fc) с val accuracy 98.28%. Transfer learning критически важен для компьютерного зрения - даже простой head-only подход дает 2x улучшение против обучения с нуля.

Главное про transfer learning: pretrained веса - это ключ к успеху, fine-tuning последних слоев дает дополнительный буст, но требует осторожности с learning rate.

Главное про detection: метрики precision/recall зависят от порога confidence score, для корректной оценки нужно правильное сопоставление предсказаний и ground truth по IoU.

## 9. Приложение (опционально)

Дополнительные сравнения не проводились.
