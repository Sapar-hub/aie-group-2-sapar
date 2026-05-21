# Итоговый проект по курсу «Инженерия Искусственного Интеллекта»

Проект: детекция штампов ГОСТ на чертежах с использованием гибридного подхода (YOLO + Faster R-CNN + DETR + OpenCV).

---

## 1. Паспорт проекта

- **Название проекта:** `Детекция штампов ГОСТ на строительных чертежах (гибридный подход YOLO + CV)`
- **Автор:** `Сапармурадов Сапармурат`
- **Группа:** `БФБО-01-24`
- **Контакт:** `saparmurat.saparmuradov@mail.ru`
- **Телеграм:** `@benotlikethose5919`

- **Краткое описание:**
> Проект реализует систему детекции штампов ГОСТ на строительных чертежах. Сравниваются 4 подхода:
> 1. CV-базовая линия (OpenCV: контуры, template matching)
> 2. YOLOv8/v11 (одностадийный детектор)
> 3. Faster R-CNN (двухстадийный детектор)
> 4. DETR (трансформерный детектор)
>
> Лучшие CNN-подходы комбинируются с CV-фильтрацией (гибрид). Результат – FastAPI сервис с `/predict` endpoint (загрузка чертежа → bbox штампа).

---

## 2. Структура проекта

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
│   ├── images/train/             # 500 синтетических (генерируются)
│   ├── labels/test/              # 49 YOLO-разметок
│   ├── labels/train/             # 500 синтетических разметок
│   └── gost_stamp.yaml           # YOLO dataset config
├── notebooks/
│   ├── exp01_eda_baseline.ipynb        # EDA: PPI, размеры штампов
│   ├── exp02_synthetic_data.ipynb      # Синтетические данные
│   ├── exp03_cv_baseline.ipynb         # CV baseline (contours)
│   ├── exp04_yolo_experiments.ipynb    # YOLOv8 train/eval
│   ├── exp05_faster_rcnn.ipynb         # Faster R-CNN train/eval
│   ├── exp06_hybrid.ipynb              # CNN + CV refine
│   └── exp07_comparison.ipynb          # Финальное сравнение
├── src/
│   ├── data/
│   │   ├── loader.py             # Загрузка изображений и разметок
│   │   ├── augment.py            # Аугментации
│   │   ├── synthetic.py          # Генерация синтетики
│   │   └── ppi.py                # PPI detection (metadata + ISO 216)
│   ├── models/
│   │   ├── cv_baseline.py        # OpenCV пайплайн
│   │   ├── yolo_model.py         # YOLO обёртка
│   │   ├── rcnn_model.py         # Faster R-CNN обёртка
│   │   └── __init__.py
│   ├── evaluation/
│   │   └── metrics.py            # IoU, precision, recall, F1
│   ├── hybrid/
│   │   ├── refiner.py            # CV-уточнение CNN-кандидатов
│   │   └── __init__.py
│   └── api/
│       ├── main.py               # FastAPI приложение + /predict
│       ├── schemas.py            # Pydantic схемы
│       └── __init__.py
├── tests/
│   ├── test_metrics.py           # 10 тестов метрик
│   └── test_models.py            # 5 тестов моделей
├── artifacts/
│   ├── models/                   # Обученные веса
│   ├── figures/                  # Графики
│   └── metrics/                  # Результаты экспериментов
└── scripts/
    ├── generate_synthetic.py
    ├── train_all.py
    ├── evaluate_all.py
    └── run_api.py
```

---

## 3. Требования и установка

### 3.1. Требования

- Python `>= 3.10`
- PyTorch `>= 2.0` (с CUDA, если доступен)
- ultralytics (YOLO)
- transformers + torchvision (Faster R-CNN, DETR)
- OpenCV, numpy, matplotlib, scikit-learn, pandas
- FastAPI, uvicorn, python-multipart

### 3.2. Установка окружения

```bash
cd project
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. Как запустить проект

### 4.1. Генерация синтетических данных

```bash
python -m scripts.generate_synthetic
```

### 4.2. Обучение моделей (Google Colab)

Все эксперименты запускаются в Google Colab (T4 GPU, Free). 

Последовательность:
1. Открыть `notebooks/exp04_yolo_experiments.ipynb` в Colab
2. Запустить все ячейки → YOLO обучение (~3-5 мин)
3. Открыть `notebooks/exp05_faster_rcnn.ipynb` в Colab
4. Запустить все ячейки → Faster R-CNN (~15-30 мин)
5. `notebooks/exp06_hybrid.ipynb` → Hybrid (~2 мин)
6. `notebooks/exp07_comparison.ipynb` → Финальное сравнение

Результаты автосохраняются в `artifacts/metrics/` и пушатся в git.

### 4.3. Запуск сервиса (локально после обучения)

```bash
python -m src.api.main
```

Сервис поднимается на `http://localhost:8000`, Swagger UI на `/docs`.

**Эндпоинты:**
- `GET /health` – проверка работоспособности
- `POST /predict` – загрузка изображения, возвращает bbox штампа и уверенность

Пример:
```bash
curl -X POST -F "file=@чертеж.png" http://localhost:8000/predict
```

---

## 5. Данные

- 49 изображений строительных чертежей с разметкой штампов в формате YOLO
- Источник: `Sapar-hub/gost-ocr`
- Синтетические данные: вырезанные штампы накладываются на случайные участки чертежей (500+ изображений)

---

## 6. Тесты

```bash
cd project
pytest tests -v
```

Результат: 15 passed, 1 skipped (RCNN пропущен — требует torch/torchvision).

---

## 7. Демонстрация на защите

1. Покажу структуру проекта и ноутбуки с экспериментами
2. Запущу сервис, отправлю чертёж через Swagger UI → получу bbox
3. Покажу таблицу сравнения 4 подходов + гибридов по IoU
4. Объясню, почему гибрид победил (статистический тест)

---

## 8. Ограничения и дальнейшая работа

- Модели обучаются на 49 изображениях + синтетике — данные ограничены
- DETR может не сойтись без большого объёма данных
- В дальнейшем: больше данных, сегментация штампов, OCR текста в штампах
