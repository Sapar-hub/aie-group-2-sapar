# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

Проект: детекция штампов ГОСТ на чертежах с использованием гибридного подхода (YOLO + Faster R-CNN + OpenCV).

---

## 1. Паспорт проекта

- **Название проекта:** Детекция штампов ГОСТ на строительных чертежах (гибридный подход YOLO + CV)
- **Автор:** Сапармурадов Сапармурат
- **Группа:** БФБО-01-24
- **Контакт:** saparmurat.saparmuradov@mail.ru
- **Телеграм:** @benotlikethose5919

**Краткое описание:**
> Система детекции штампов ГОСТ на строительных чертежах. Сравниваются 4 подхода:
> 1. CV-базовая линия (OpenCV: контуры, template matching)
> 2. YOLOv8n (одностадийный детектор)
> 3. Faster R-CNN (двухстадийный детектор)
> 4. Hybrid (YOLO + CV уточнение)
>
> Лучший подход развёрнут как FastAPI сервис с `/predict` endpoint.

---

## 2. Предметная область и постановка задачи

### 2.1. Домен и пользователь

**Область:** Автоматизация анализа строительной документации. Штампы ГОСТ — обязательный элемент каждого чертежа (содержат номер проекта, дату, масштаб, лист). Инженеры и архитекторы тратят время на ручной поиск штампа перед OCR-распознаванием.

**Пользователь:** Инженер/архитектор, оцифровывающий пачку чертежей. Сервис автоматически находит штамп, чтобы передать его в OCR.

### 2.2. Задача в терминах ML

- **Тип:** Детекция одного класса (`stamp`) на изображениях чертежей
- **Вход:** RGB-изображение чертежа формата A1–A4 (200–400 DPI)
- **Выход:** Bounding box штампа + confidence
- **Стандарт:** Штамп ГОСТ — 185×55 мм, aspect ratio ≈ 3.55:1, всегда в правом нижнем углу

### 2.3. Целевые метрики

| Метрика | Зачем |
|---------|-------|
| **IoU** | Точность локализации — насколько bbox совпадает с настоящим штампом. Основная метрика. |
| **Precision / Recall / F1** | Качество детекции при пороге IoU ≥ 0.5. |
| **Detection rate** | Доля изображений, на которых модель нашла штамп. |
| **Время инференса** | Практическое ограничение для сервиса. |

Выбор финальной модели — по балансу F1 и скорости.

---

## 3. Структура проекта

```
project/
├── README.md                     # Этот файл
├── plan.md                       # План работ
├── requirements.txt              # Зависимости
├── report.md                     # Отчёт
├── self-checklist.md             # Чеклист
├── pyproject.toml                # Упаковка Python
├── configs/
│   ├── config.yaml               # Основной конфиг
│   └── .env.example              # Шаблон секретов
├── data/
│   ├── images/test/              # 49 реальных изображений
│   ├── images/train/             # 500 синтетических
│   ├── labels/test/              # 49 YOLO-разметок
│   ├── labels/train/             # 500 синтетических разметок
│   └── gost_stamp.yaml           # YOLO dataset config
├── notebooks/
│   ├── exp01_eda_baseline.ipynb
│   ├── exp02_synthetic_data.ipynb
│   ├── exp03_cv_baseline.ipynb
│   ├── exp04_yolo_experiments.ipynb
│   ├── exp05_faster_rcnn.ipynb
│   ├── exp06_hybrid.ipynb
│   └── exp07_comparison.ipynb
├── src/
│   ├── data/         # loader, augment, synthetic, image_quality
│   ├── models/       # cv_baseline, yolo_model, rcnn_model
│   ├── evaluation/   # metrics (IoU, precision, recall, F1)
│   ├── hybrid/       # refiner (CNN → CV refine)
│   └── api/          # main.py (FastAPI), schemas.py
├── tests/
│   ├── test_metrics.py           # 10 тестов метрик
│   └── test_models.py            # 6 тестов моделей
├── artifacts/
│   ├── models/                   # Веса (фактически в yolo/exp01/)
│   ├── figures/                  # 7 графиков
│   ├── metrics/                  # EDA summary, YOLO results
│   └── yolo/exp04/weights/       # best.pt (6 MB, 50 эпох)
└── scripts/
    ├── generate_synthetic.py
    └── train_all.py
```

---

## 4. Требования и установка

### 4.1. Требования

- Python `>= 3.10`
- PyTorch `>= 2.0` (с CUDA, если доступен)
- ultralytics (YOLO)
- torchvision (Faster R-CNN)
- OpenCV, numpy, matplotlib, scikit-learn, pandas
- FastAPI, uvicorn, python-multipart

### 4.2. Установка окружения

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 4.3. Генерация синтетических данных

```bash
python -m scripts.generate_synthetic
```

### 4.4. Обучение моделей (Google Colab)

Все эксперименты запускаются в Google Colab (T4 GPU, Free).

Последовательность:
1. Открыть `notebooks/exp04_yolo_experiments.ipynb` в Colab
2. Запустить все ячейки → YOLO обучение (~3-5 мин)
3. Открыть `notebooks/exp05_faster_rcnn.ipynb` в Colab
4. Запустить все ячейки → Faster R-CNN (~15-30 мин)
5. `notebooks/exp06_hybrid.ipynb` → Hybrid (~2 мин)
6. `notebooks/exp07_comparison.ipynb` → Финальное сравнение

Результаты автосохраняются в `artifacts/metrics/`.

### 4.5. Запуск сервиса

Перед запуском скопируйте веса YOLO в ожидаемую директорию:

```bash
cp artifacts/yolo/exp04/weights/best.pt artifacts/models/best.pt
python -m src.api.main
```

Сервис на `http://localhost:8000`, Swagger UI на `/docs`.

**Эндпоинты:**
- `GET /health` — проверка работоспособности
- `POST /predict` — загрузка изображения, возвращает bbox штампа и уверенность

Пример:
```bash
curl -X POST -F "file=@чертеж.png" http://localhost:8000/predict
```

### 4.6. Тестирование

```bash
cd project
pytest tests -v
```

**Результат:** 15 passed, 1 skipped (RCNN — требует torch/torchvision)

---

## 5. Результаты экспериментов

| Модель | Версия данных | Выборка | IoU mean | Precision | Recall | F1 | Detection rate |
|--------|---------------|---------|----------|-----------|--------|-----|---------------|
| CV Baseline | v3 (обновлён) | 35 real (non-val, non-donor) | 0.633 | 0.800 | 0.686 | 0.738 | 85.7% |
| YOLOv8n | v3 (clean) | 35 real (non-val, non-donor) | **0.700** | **1.000** | 0.778 | **0.875** | 77.8% |
| Faster R-CNN | v2* | 45 real (non-donor) | **0.731** | 0.949 | **0.822** | **0.881** | **86.7%** |
| Hybrid (YOLO+CV) | — | — | — | — | — | — | — |

> **\*** Faster R-CNN v2 оценён на 45 изображениях (включает 10 val, не исключённых из теста) — не полностью сопоставим со строками v3. Подробнее об утечках данных: [data/LEAKAGE.md](data/LEAKAGE.md).

---

## 6. Данные

- 49 реальных изображений строительных чертежей с разметкой штампов в формате YOLO
- Источники: `https://2d-3d.ru/`
- 500 синтетических изображений (250 GOST + 250 copy-paste)
- **Чистый baseline (v3):** 500 синтетических на train, 10 реальных non-donor на val (`val_honest`), тест на 35 non-donor, non-val
- Предотвращение утечки: штампы вырезаются только из 4 изображений-доноров; фон copy-paste — из 20 независимых изображений
- История утечек (v1/v2/v3) задокументирована в [data/LEAKAGE.md](data/LEAKAGE.md) и [data/README.md](data/README.md)

---

## 7. Ограничения и дальнейшая работа

- Модели обучаются на 49 изображениях + синтетике — данные ограничены
- DETR не реализован (stretch goal)
- В дальнейшем: больше данных, сегментация штампов, OCR текста в штампах
