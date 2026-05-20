# Plan: Stamp Detection — Hybrid YOLO + CV

## Goal

Build and compare 4 approaches for GOST stamp detection on construction drawings, then combine the best CNN with CV refinement into a hybrid. Deploy as a FastAPI service.

---

## Approaches

| # | Approach | Type | Training Data |
|---|---|---|---|
| 1 | **CV Baseline** (contours, template matching, color thresholding) | Classical | None |
| 2 | **YOLOv8/v11** | One-stage CNN | 49 real + synthetic |
| 3 | **Faster R-CNN** | Two-stage CNN | 49 real + synthetic |
| 4 | **DETR** | Transformer detector | 49 real + synthetic |
| 5 | **Hybrid** (best CNN → CV refine) | Combined | Best config from above |

## Experiments

Each CNN approach has 3 variants:
- **No augmentation** — train on 49 images only
- **Augmentation only** — train on 49 images with augmentations
- **Augmentation + synthetic** — train on 49 + 500+ synthetic images

Plus hybrid variants for each best CNN config.

Total: **~16 experiment configurations** × 5 folds = ~80 training runs.

---

## Protocol

- **Cross-validation:** 5-fold on the 49 labeled images
- **Synthetic generation:** Crop stamp regions → paste onto non-stamp drawing regions with random transforms (rotation, scale, brightness)
- **Augmentations:** Horizontal flip, rotation ±15°, scale jitter, brightness/contrast, cutout
- **Primary metric:** IoU (Intersection over Union)
- **Secondary metrics:** Precision, Recall, F1, inference time
- **Statistical test:** Paired Wilcoxon across folds to justify model selection

---

## Implementation Order

### Phase 1: Data Pipeline
- [x] `src/data/loader.py` — load images + YOLO labels, visualize
- [x] `src/data/augment.py` — augmentation transforms
- [x] `src/data/synthetic.py` — synthetic data generation pipeline (GOST + copy-paste)
- [x] `scripts/generate_synthetic.py` — CLI to generate dataset
- [x] `notebooks/exp01_eda_baseline.ipynb` — EDA: image stats, label distribution, PPI analysis
- [x] `notebooks/exp02_synthetic_data.ipynb` — synthetic data exploration

### Phase 2: CV Baseline
- [ ] `src/models/cv_baseline.py` — contour detection, template matching, color-based
- [ ] `notebooks/02_cv_baseline.ipynb` — results, parameter sweeps, failure analysis

### Phase 3: YOLO Experiments
- [ ] `src/models/yolo_model.py` — train/inference wrapper
- [ ] `notebooks/03_yolo_experiments.ipynb` — 3 variants, 5-fold CV results

### Phase 4: Faster R-CNN Experiments
- [ ] `src/models/rcnn_model.py` — train/inference wrapper
- [ ] `notebooks/04_faster_rcnn.ipynb` — 3 variants, 5-fold CV results

### Phase 5: DETR Experiments (stretch)
- [ ] `src/models/detr_model.py` — train/inference wrapper
- [ ] `notebooks/05_detr.ipynb` — 3 variants, 5-fold CV results

### Phase 6: Hybrid
- [ ] `src/hybrid/refiner.py` — CV post-processing of CNN proposals
- [ ] `notebooks/06_hybrid.ipynb` — all hybrid variants

### Phase 7: Comparison
- [ ] `notebooks/07_comparison.ipynb` — aggregate metrics, statistical tests, plots
- [ ] `src/evaluation/metrics.py` — reusable metric computation

### Phase 8: API Service
- [ ] `src/api/main.py` — FastAPI app
- [ ] `src/api/schemas.py` — Pydantic request/response
- [ ] `src/api/router.py` — /predict endpoint

### Phase 9: Project Infrastructure
- [ ] `configs/config.yaml` — all paths, model params, experiment config
- [ ] `configs/.env.example` — template for secrets
- [ ] `requirements.txt` — pinned dependencies
- [ ] `pyproject.toml` — package metadata + CLI entry points
- [ ] `tests/test_metrics.py` — IoU calculation tests
- [ ] `tests/test_pipeline.py` — end-to-end sanity checks

### Phase 10: Documentation
- [ ] `report.md` — fill with results and analysis
- [ ] `self-checklist.md` — complete self-check
- [ ] `README.md` — updated with final commands

---

---

## PPI (Pixels Per Inch) Detection

### Terminology
- **DPI** (Dots Per Inch) — scanner setting, used when printing
- **PPI** (Pixels Per Inch) — digital image property
- For scanned technical drawings: PPI = DPI (these terms are often used interchangeably, but PPI is more accurate for digital)

### Detection Logic (priority order)

1. **Metadata extraction** (via `PIL.Image.info['dpi']`)
   - If present and reasonable → use it
   - Scanner settings are usually correct for full-page scans

2. **ISO 216 inference** (dimension-based)
   - Match image dimensions to standard A-series paper sizes (A1, A2, A3, A4)
   - Use aspect ratio √2 (≈1.414) to identify paper size
   - Calculate: `Spacing (mm/pixel) = Physical Width (mm) / Pixel Width`

   ```
   A4: 210×297 mm
   A3: 297×420 mm
   A2: 420×594 mm
   A1: 594×841 mm
   ```

3. **Contradiction handling**
   - Metadata exists + dimensions suggest different PPI → check tolerance (±10%)
   - If contradiction → trust inference (dimensions are more "honest" after resize/compression)
   - Validate via stamp size: stamp in mm should be consistent across images

### Validation in EDA

`notebooks/01_data_exploration.ipynb` should include:

1. **PPI Distribution** — how many images have metadata vs inference
2. **Contradiction Analysis** — when do they conflict?
3. **Accuracy Check** — does stamp size in mm stay consistent?
4. **Edge Cases** — non-standard paper sizes, cropped images

---

## Risks

| Risk | Mitigation |
|---|---|
| DETR won't converge on 49 images | Make DETR a stretch goal; drop if results are worse than random |
| Synthetic data looks unrealistic | Randomize background patches; use multiple non-stamp regions |
| 80 training runs is time-consuming | YOLO is fast (minutes/run); R-CNN moderate; DETR slowest. Run overnight. |
| IoU metric is noisy with small data | 5-fold CV gives 5 estimates per config; use Wilcoxon for claims |
| PPI metadata contradictory | Use dimension-based inference as fallback, validate via stamp size consistency |
