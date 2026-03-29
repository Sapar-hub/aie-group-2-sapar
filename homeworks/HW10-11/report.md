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

- Таблица результатов: `artifacts/runs.csv`
- Лучшая модель части A: `artifacts/best_classifier.pt`
- Конфиг лучшей модели части A: `artifacts/best_classifier_config.json`
- Кривые лучшего прогона классификации: `artifacts/figures/classification_curves_best.png`
- Сравнение C1-C4: `artifacts/figures/classification_compare.png`
- Визуализация аугментаций: `artifacts/figures/augmentations_preview.png`
- Визуализации второй части: `artifacts/figures/detection_examples.png`, `artifacts/figures/detection_metrics.png`

Короткая сводка:

- Лучший эксперимент части A: **C3 (ResNet18 head-only)**
- Лучшая val_accuracy: **0.926**
- Итоговая test_accuracy лучшего классификатора: **0.92675**
- Что дали аугментации (C2 vs C1): Аугментации не помогли - C2 (0.391) хуже C1 (0.461), возможно из-за слишком агрессивных аугментаций.
- Что дал transfer learning (C3/C4 vs C1/C2): Огромное улучшение! C3 (0.926) и C4 (0.918) значительно лучше простых CNN.
- Что оказалось лучше: head-only или partial fine-tuning: **Head-only (C3) оказался немного лучше partial fine-tuning (C4) на val (0.926 vs 0.918)**, что может быть связано с малым числом эпох обучения.
- Что показал режим V1 во второй части: score_threshold=0.3 дает **высокий recall (0.90) и низкий precision (0.27)**.
- Что показал режим V2 во второй части: score_threshold=0.7 — более строгий фильтр, **precision вырос до 0.52, а recall незначительно снизился до 0.86**.
- Как интерпретируются метрики второй части: Метрики демонстрируют классический trade-off между precision и recall.

## 7. Анализ

Простая CNN (C1) показала умеренные результаты (46.1% val accuracy). Аугментации (C2) в данном случае ухудшили результат до 39.1%, что может говорить о том, что для данного датасета и малого числа эпох они были избыточны или слишком агрессивны.

Transfer learning с pretrained ResNet18 дал огромный прирост: C3 (head-only) достиг **92.6%** val accuracy, а C4 (partial fine-tune layer4+fc) — **91.8%**. В данном случае, **head-only оказался эффективнее**. Вероятно, это связано с тем, что двух эпох было недостаточно для стабильного и качественного дообучения более глубоких слоев в C4, в то время как обучение только классификационной головы является более простой задачей и быстрее сходится к хорошему результату.

Во второй части (detection) эксперименты V1 и V2 наглядно показали зависимость метрик от порога уверенности. При score_threshold=0.3 (V1) модель генерирует много предсказаний, что приводит к **высокому recall (90.4%)**, но страдает **precision (27.3%)** из-за большого числа ложных срабатываний. Увеличение порога до 0.7 в V2 отсекает неуверенные предсказания, что **существенно увеличивает precision до 52.2%** ценой небольшого падения **recall до 86.0%**.

## 8. Итоговый вывод

Для задачи классификации на STL10 в рамках данного эксперимента (2 эпохи) оптимальным оказался подход **C3 (ResNet18 head-only) с val accuracy 92.6%**. Transfer learning является ключевым для достижения высокой точности, однако стратегия fine-tuning требует более аккуратной настройки и, возможно, большего числа эпох для демонстрации своего преимущества над head-only подходом.

Главное про detection: метрики precision/recall **сильно** зависят от порога confidence score. Выбор этого порога является компромиссом между полнотой обнаружения (recall) и точностью предсказаний (precision), который определяется требованиями конкретной задачи.

## 9. Приложение (опционально)

Дополнительные сравнения не проводились.
