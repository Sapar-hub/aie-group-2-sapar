# AI Engineering Coursework — Saparmyradov Saparmyrat

**Group:** БФБО-01-24 · **Contact:** @benotlikethose5919 · saparmurat.saparmuradov@mail.ru

---

This repository contains my coursework for the **AI Engineering** (Инженерия Искусственного Интеллекта) program — spanning Python fundamentals through to modern deep learning, computer vision, NLP, and retrieval-augmented generation.

---

## Final Project: GOST Stamp Detection on Construction Drawings

Detects GOST-standard stamps on engineering drawings using a hybrid deep learning + computer vision approach. Four methods were compared:

| Approach | IoU | F1 | Detection Rate |
|----------|-----|-----|---------------|
| CV Baseline (OpenCV contours) | 0.330 | 0.490 | 49% |
| **YOLOv8n** | **0.880** | **1.000** | **100%** |
| Faster R-CNN | — | — | ⏳ Pending |
| Hybrid (YOLO + CV refine) | — | — | ⏳ Pending |

YOLOv8n achieves **IoU 0.88** with **100% detection rate** and is deployed as a **FastAPI** service at `/predict`.

📁 [`project/`](project/) — code, notebooks, API, report

---

## Coursework Journey

### Fundamentals & Data Engineering

| HW | Topic | What I Did | Tools |
|----|-------|------------|-------|
| **HW01** | NumPy basics | Array creation, vectorized operations, memory efficiency | NumPy |
| **HW02** | EDA with Pandas | Data loading, grouping, aggregations, histograms & boxplots | Pandas, Seaborn |
| **HW03** | CLI EDA tool | Built `eda-cli` — a CLI that generates EDA reports from CSVs | Click, Pandas, Matplotlib |
| **HW04** | HTTP EDA service | Wrapped `eda-cli` as a FastAPI service with data quality heuristics | FastAPI, Uvicorn |

### Classical Machine Learning

| HW | Topic | What I Did | Tools |
|----|-------|------------|-------|
| **HW05** | Linear models & fair ML protocol | Stratified split, Logistic Regression, avoiding data leakage | scikit-learn |
| **HW06** | Ensemble methods | Compared 6 classifiers (Dummy → Stacking), GridSearchCV, ROC-AUC **0.911** | scikit-learn |
| **HW07** | Clustering | KMeans vs DBSCAN on 3 datasets (different scales, nonlinear, varying density) | scikit-learn, PCA |

### Deep Learning

| HW | Topic | What I Did | Tools |
|----|-------|------------|-------|
| **HW08-09** | MLP regularization & optimization | Dropout, BatchNorm, EarlyStopping, SGD vs Adam on EMNIST (47 classes, **85% val acc**) | PyTorch |
| **HW10-11** | CNN, transfer learning, object detection | Custom CNN → ResNet18 head-only (**92.7% test**), Faster R-CNN on Pascal VOC | Torchvision, Albumentations |
| **HW12** | Time series forecasting | GRU (MAE 4.97) vs naive/moving-average/Ridge baselines, temporal split | PyTorch, scikit-learn |

### NLP & Retrieval

| HW | Topic | What I Did | Tools |
|----|-------|------------|-------|
| **HW13** | BERT fine-tuning | DistilBERT for 6-class emotion classification, **93% accuracy**, confusion matrix analysis | Hugging Face Transformers |
| **HW14** | Embeddings, FAISS, Mini-RAG | FAISS index (hit@3=1.0), chunking experiments, RAG pipeline with TF-IDF sentence ranking + citations | FAISS, Sentence-Transformers |

---

## Technology Stack

**Languages:** Python  
**ML/DL:** PyTorch, Torchvision, Ultralytics (YOLO), Hugging Face Transformers, scikit-learn  
**CV:** OpenCV, Albumentations  
**NLP/Retrieval:** Sentence-Transformers, FAISS, TF-IDF  
**Data:** NumPy, Pandas, Matplotlib, Seaborn  
**Infra:** FastAPI, Uvicorn, Click (CLI), Pytest, uv  

---

*Detailed reports in each `homeworks/HW*/report.md` and `project/report.md`.*
