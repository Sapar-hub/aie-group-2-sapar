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
│   ├── images/test/              # 49 тестовых изображений
│   ├── synthetic/                # Синтетические данные
│   └── labels/test/              # 49 YOLO-разметок
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_cv_baseline.ipynb
│   ├── 03_yolo_experiments.ipynb
│   ├── 04_faster_rcnn.ipynb
│   ├── 05_detr.ipynb
│   ├── 06_hybrid.ipynb
│   └── 07_comparison.ipynb
├── src/
│   ├── data/
│   │   ├── loader.py             # Загрузка изображений и разметок
│   │   ├── augment.py            # Аугментации
│   │   └── synthetic.py          # Генерация синтетики
│   ├── models/
│   │   ├── cv_baseline.py        # OpenCV пайплайн
│   │   ├── yolo_model.py         # YOLO обёртка
│   │   ├── rcnn_model.py         # Faster R-CNN обёртка
│   │   └── detr_model.py         # DETR обёртка
│   ├── evaluation/
│   │   └── metrics.py            # IoU, precision, recall, F1
│   ├── hybrid/
│   │   └── refiner.py            # CV-уточнение CNN-кандидатов
│   └── api/
│       ├── main.py               # FastAPI приложение
│       ├── schemas.py            # Pydantic схемы
│       └── router.py             # /predict endpoint
├── tests/
│   ├── test_metrics.py
│   └── test_models.py
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

### 4.2. Обучение всех моделей

```bash
python -m scripts.train_all
```

### 4.3. Оценка всех моделей

```bash
python -m scripts.evaluate_all
```

### 4.4. Запуск сервиса

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
pytest tests
```

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
