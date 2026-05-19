# Migration Plan: GOST-OCR to Current Directory

## Goal
Migrate GOST-OCR project to fulfill all 10 criteria from `self-checklist.md`.

---

## Phase 1: Project Setup (Files & Structure)

### 1.1 Copy Core Project Structure
- [ ] Copy `src/gost_ocr/` module from `~/playground/metrogiprotrans/src/gost_ocr/` → `./src/gost_ocr/`
- [ ] Copy `pyproject.toml` → root
- [ ] Copy `.python-version` → root
- [ ] Copy `AGENTS.md` → root (reference for commands)

### 1.2 Create Missing Required Directories
- [ ] Create `notebooks/` directory
- [ ] Create `configs/` directory
- [ ] Create `tests/` directory
- [ ] Create `artifacts/` directory
- [ ] Create `data/` with sample data

### 1.3 Create Missing Required Files
- [ ] Create `report.md` (template: see course requirements)
- [ ] Create `self-checklist.md` (10 criteria, mark as TBD)
- [ ] Create `.env.example` (API keys placeholders)
- [ ] Create `Dockerfile` (containerize the service)
- [ ] Copy `yolov8n.pt` → `artifacts/` (base model)
- [ ] Copy trained `best.pt` → `artifacts/` (if available)

---

## Phase 2: Code Adaptation (Service Layer)

### 2.1 Convert CLI to HTTP API
- [ ] Add FastAPI service in `src/service/` (not just CLI)
- [ ] Implement `/predict` endpoint (uses real YOLO model)
- [ ] Implement `/health` endpoint
- [ ] Add logging throughout

### 2.2 Refactor Entry Points
- [ ] Keep CLI for batch processing (existing)
- [ ] Add HTTP API mode (new)
- [ ] Add unified `main.py` or `__main__.py`

### 2.3 Configuration Management
- [ ] Move hardcoded paths to `configs/config.yaml`
- [ ] Add `.env` loading for secrets
- [ ] Document config options

---

## Phase 3: EDA & Experiments (Notebooks)

### 3.1 Create EDA Notebook
- [ ] `notebooks/01_eda.ipynb`
  - Load sample images
  - Visualize stamp position distribution (top/bottom/left/right)
  - Analyze image dimensions, DPI distribution
  - Show example drawings with detected stamps
  - Statistics on stamp sizes

### 3.2 Create Baseline Experiments
- [ ] `notebooks/02_baseline.ipynb`
  - Run OpenCV detection (baseline method)
  - Run YOLO detection (learned model)
  - Compare IoU metrics on test set
  - Document results

### 3.3 Create Model Comparison
- [ ] `notebooks/03_model_comparison.ipynb`
  - Compare YOLO variants (if training feasible)
  - Compare YOLO vs EasyOCR as OCR baseline
  - Present metrics: IoU, CER, WER

---

## Phase 4: Documentation

### 4.1 Update README.md
- [ ] Rewrite for GOST-OCR project
- [ ] Add "How to run" section for API
- [ ] Add demo scenario for defense
- [ ] Document dependencies (uv/pip)

### 4.2 Write report.md
- [ ] Problem statement
- [ ] Data description
- [ ] EDA findings
- [ ] Methodology (YOLO, OpenCV)
- [ ] Experiments & metrics
- [ ] Model selection justification
- [ ] Limitations & future work

### 4.3 Complete self-checklist.md
- [ ] Fill in actual file locations for each criterion

---

## Phase 5: Testing & Verification

### 5.1 Add Basic Tests
- [ ] `tests/test_preprocessing.py`
- [ ] `tests/test_detection.py`
- [ ] `tests/test_api.py`

### 5.2 Verify Against Criteria
- [ ] Test service starts from README
- [ ] Verify `/predict` uses real model
- [ ] Verify notebooks exist and run
- [ ] Check code structure (src/ organization)
- [ ] Verify Dockerfile builds

---

## Phase 6: Data (If Needed)

### 6.1 Expand Training Data
- [ ] Assess if 25 images is enough
- [ ] Add more synthetic technical drawings if needed
- [ ] Document data sources in `data/README.md`

---

## Estimated Timeline

| Phase | Task | Notes |
|-------|------|-------|
| 1 | Setup | ~30 min |
| 2 | Code Adaptation | ~2-3 hours |
| 3 | Notebooks | ~2 hours |
| 4 | Documentation | ~1-2 hours |
| 5 | Testing | ~30 min |
| 6 | Data (optional) | If needed |

---

## Success Criteria

After migration, project should have:
- ✅ Working HTTP API with `/predict` and `/health`
- ✅ Real YOLO model loaded in service
- ✅ EDA notebook with visualizations
- ✅ Model comparison notebook with metrics
- ✅ Clean src/ structure
- ✅ Dockerfile
- ✅ .env.example (no real secrets)
- ✅ Logging + observability
- ✅ report.md with model justification
- ✅ Demo scenario in README

**Target: 9-10 checklist points → Grade 5**