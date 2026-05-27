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
    ├── Dockerfile                    # Контейнеризация (multi-stage)
    ├── .dockerignore                 # Исключения для Docker build
    ├── docker-compose.yml            # Оркестрация с healthcheck
    ├── configs/
    │   ├── config.yaml               # Основной конфиг
    │   └── .env.example              # Шаблон секретов
├── data/
│   ├── images/test/              # 49 реальных изображений
│   ├── images/train_v4/          # 500 синтетических
│   ├── labels/test/              # 49 YOLO-разметок
│   ├── labels/train_v4/          # 500 синтетических разметок
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
│   ├── data/         # loader (+ GOSTDataset для torchvision), synthetic, image_quality
│   ├── models/       # cv_baseline, train_yolo, rcnn_model
│   ├── evaluation/   # metrics (IoU, precision, recall, F1)
│   ├── hybrid/       # refiner (CNN → CV refine)
│   └── api/          # main.py (FastAPI), schemas.py
├── tests/
│   ├── test_metrics.py           # 10 тестов метрик
│   └── test_models.py            # 6 тестов моделей
├── artifacts/
│   ├── models/                   # SYMLINK → yolo/v4/weights/best.pt
│   ├── yolo/v4/weights/          # best.pt (6 MB, 50 эпох, v4 current)
│   ├── yolo/legacy/              # v3, v2, v1, rcnn_v2 (исторические)
│   ├── rcnn/v4/                  # rcnn_best.pth (Faster R-CNN v4)
│   ├── figures/                  # графики (v4 + cross-model)
│   └── metrics/                  # EDA summary, v4 results, comparison
└── scripts/
    ├── generate_synthetic.py      # CLI: генерация синтетического датасета
    └── train_all.py               # CLI: обучение YOLO + Faster R-CNN
```

---

## 4. Требования и установка

### 4.1. Требования

- Python `>= 3.10`
- git-lfs (для скачивания весов моделей)
- PyTorch `>= 2.0` (с CUDA, если доступен)
- ultralytics (YOLO)
- torchvision (Faster R-CNN)
- OpenCV, numpy, matplotlib, scikit-learn, pandas
- FastAPI, uvicorn, python-multipart

### 4.2. Установка окружения

> **Веса моделей хранятся в Git LFS.** При клонировании репозитория на новой машине обязательно скачайте их перед запуском:

```bash
git lfs install
git lfs pull

cd project
uv venv
source .venv/bin/activate
uv sync
```

### 4.3. Генерация синтетических данных **(опционально)**

```bash
python -m scripts.generate_synthetic
```

### 4.4. Обучение моделей **(опционально)**

**Вариант A — CLI (локально, CPU):**

```bash
python -m scripts.train_all
```

Обучает YOLO + Faster R-CNN последовательно. R-CNN на CPU может быть медленным (~30 мин на v4, 500 samples). Результаты и веса сохраняются в `artifacts/`.

**Вариант B — Google Colab (GPU, быстрее):**

Последовательность:
1. Открыть `notebooks/exp04_yolo_experiments.ipynb` в Colab
2. Запустить все ячейки → YOLO обучение (~3-5 мин)
3. Открыть `notebooks/exp05_faster_rcnn.ipynb` в Colab
4. Запустить все ячейки → Faster R-CNN (~15-30 мин)
5. `notebooks/exp06_hybrid.ipynb` → Hybrid (~2 мин)
6. `notebooks/exp07_comparison.ipynb` → Финальное сравнение

Результаты автосохраняются в `artifacts/metrics/`.

### 4.5. Запуск сервиса

Перед запуском убедитесь что symlink не битый, ведет на правильный путь и обновите на актуальную модель при необходимости:

```bash
ln -sf ../yolo/v4/weights/best.pt artifacts/models/best.pt
```

**Вариант A — через entry point (после `uv sync`):**
```bash
gost-detect
```

**Вариант B — напрямую:**
```bash
python -m src.api.main
```

**Вариант C — Docker (для сервера):**
```bash
docker compose build
docker compose up -d
```

Сервис будет доступен на `http://localhost:8000`, Swagger UI на `/docs`.

Переключение на другую версию (например, v3):
```bash
ln -sf ../yolo/legacy/v3/weights/best.pt artifacts/models/best.pt
```

Сервис на `http://localhost:8000`, корень `/` сразу перенаправляет на `/docs`.

**Эндпоинты:**
- `GET /` — редирект на Swagger UI (`/docs`)
- `GET /health` — проверка работоспособности
- `POST /predict` — загрузка изображения, возвращает bbox и уверенность в JSON
- `POST /predict/image` — загрузка изображения, возвращает JPG с нарисованным bbox

Пример:
```bash
curl -X POST -F "file=@чертеж.png" http://localhost:8000/predict
curl -X POST -F "file=@чертеж.png" http://localhost:8000/predict/image > result.jpg
```

### 4.6. Тестирование

```bash
cd project
uv run pytest tests -v
```

**Результат:** 18 passed (при наличии torch/torchvision)

---

## 5. Результаты экспериментов

| Модель | Версия данных | Выборка | IoU | Prec | Recall | F1 | Det% |
|--------|---------------|---------|-----|------|--------|-----|------|
| CV Baseline (без position map) | v4 | 35 real | 0.640 | 0.774 | 0.686 | **0.727** | 88.6% |
| YOLOv8n v3 (top-1 conf) | v3: 250 GOST + 250 CP | 35 real | 0.692 | 1.000 | 0.771 | **0.871** | 77.1% |
| Hybrid v3 (v3 + CV refine) | v3 data | 35 real | 0.692 | 0.964 | 0.771 | **0.857** | 80.0% |
| YOLOv8n v4 (top-1 conf) | v4: 50 GOST + 250 CP + 200 bg | 35 real | 0.579 | 0.885 | 0.657 | 0.754 | 74.3% |
| **Hybrid v4 (v4 + CV refine)** | v4 data | 35 real | 0.527 | 0.875 | 0.600 | **0.712** | 68.6% |
| Faster R-CNN v4 | v4 | 35 real | 0.409 | 1.000 | 0.457 | 0.627 | 45.7% |

**Версии данных:**
> - **v3** (clean baseline): 250 GOST + 250 copy-paste на независимых фонах.
>   YOLO v3 достигает F1=0.857 за счёт shortcut'а "белая сетка = штамп" — модель
>   научилась отличать синтетический фон от реального, а не детектировать штамп.
>   **v1–v3 невоспроизводимы** вне оригинальной Colab-сессии (PPI-пайплайн, отбор доноров,
>   стратификация val, RNG отличаются в разных окружениях). История версий:
>   [`data/LEAKAGE.md`](data/LEAKAGE.md).
> - **v4** (текущая, сложнее): 50 GOST + 250 copy-paste + 200 GOST-on-real-background.
>   Все актуальные оценки (YOLO v4, Hybrid v4) используют эту версию.
> - **Hybrid v4** не улучшает YOLO v4 (F1 = 0.712 vs 0.754) — CV-уточнение поверх
>   YOLO v4 снижает F1 на −0.042. Причина: сложная смесь v4 (GOST-on-real-bg) уже
>   не даёт shortcut'ов, и CV-фильтрация только теряет истинные detection'ы.
>   v3 не нуждался в уточнении — модель была насыщена за счёт shortcut'ов.

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

- 35 тестовых изображений — перекрывающиеся доверительные интервалы между моделями
- В дальнейшем: больше данных, сегментация штампов, OCR текста в штампах
