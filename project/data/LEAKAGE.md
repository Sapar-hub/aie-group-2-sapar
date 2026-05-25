# Data Leakage History & Fix

## Background

This document records the data leakage issues discovered during the development
of the GOST stamp detection pipeline, and the steps taken to fix them.

The core problem: **copy-paste synthetic data used test-set images as backgrounds**,
allowing the model to memorise background textures rather than learning stamp
features. This inflated all metrics.

---

## v1 — Contaminated

| Aspect | Detail |
|--------|--------|
| **Train** | 500 synthetic images (250 GOST + 250 copy-paste) |
| **Copy-paste backgrounds** | **Test-set images** (49 real images) |
| **Donors** | 4 stamp sources from the test set |
| **Val** | Synthetic (80/20 split of the 500) |
| **Test** | 49 real images (4 donors excluded → 45 eval) |
| **Metrics** | IoU **0.880**, Precision **1.0**, Recall **1.0**, F1 **1.0** |

**Problem:** Copy-paste pasted stamps onto backgrounds that were themselves
test-set images. The model saw test-background textures during training and
achieved unrealistically perfect precision/recall. The 1.0 metrics are a
clear sign of leakage.

**Result files:** `artifacts/metrics/yolo_v1_results.json`, `yolo_v1_runs.csv`

---

## v2 — Backgrounds fixed, val still synthetic

| Aspect | Detail |
|--------|--------|
| **Train** | 408 synthetic (204 GOST + 204 copy-paste, 80% of 500) |
| **Copy-paste backgrounds** | **20 independent images** from `data/unlabeled/` |
| **Donors** | Same 4 stamp sources from the test set |
| **Val** | 92 synthetic (46 GOST + 46 copy-paste, 20% of 500) |
| **Test** | 49 real images (4 donors excluded → 45 eval) |
| **Metrics** | IoU **0.765**, Precision **1.0**, Recall **0.844**, F1 **0.916** |

**Fix:** Backgrounds for copy-paste changed from test images to 20 independently
sourced unlabeled images from `data/unlabeled/`. Background leakage closed.

**Remaining issue:** The YOLO validation split during training (92 synthetic
images) shares the same donor stamps with the training set. This means
hyperparameter tuning through validation could overfit to donor stamp appearances.
The real test set (45 non-donor images) provides the only unbiased signal.

**Result files:** `artifacts/metrics/yolo_v2_results.json`, `yolo_v2_runs.csv`

---

## v3 — Clean baseline

| Aspect | Detail |
|--------|--------|
| **Train** | 500 synthetic (250 GOST + 250 copy-paste, **all train, no 80/20 split**) |
| **Copy-paste backgrounds** | 20 independent images from `data/unlabeled/` |
| **Donors** | Same 4 stamp sources from the test set |
| **Val** | **10 real non-donor images** (`val_honest/`, stratified selection) |
| **Test** | **35 real non-donor, non-val images** (4 donors + 10 val excluded) |
| **Status** | **Computed** — see `yolo_v3_results.json` |

**Fixes:**
1. **Copy-paste backgrounds** — already fixed since v2 (independent sources)
2. **YOLO validation** — changed from synthetic (shared donor stamps) to 10 real
   non-donor images selected via stratified sampling (PPI × sharpness × stamp size)
3. **All synthetic data for training** — removed the 80/20 split; all 500
   synthetic images used for training
4. **Test set** — reduced to 35 images (10 val + 4 donors excluded)

**Result files:** `artifacts/metrics/yolo_v3_results.json`, `yolo_v3_runs.csv`

---

## Comparison table

| Version | Train | Val (YOLO) | Test | Background leak? | Val leak? | IoU | F1 |
|---------|-------|-----------|------|-----------------|-----------|-----|-----|
| v1 | 500 synth (80% split) | 92 synth | 45 real | **YES** | no | 0.880 | 1.000 |
| v2 | 408 synth (80% split) | 92 synth | 45 real | fixed | minor* | 0.765 | 0.916 |
| v3 | 500 synth (all train) | 10 real | 35 real | fixed | fixed | **0.700** | **0.875** |

\* Minor: val shared donor stamps with train, but v2 metrics are on real test images so largely unaffected.

---

## v4 — Harder synthetic mix (current)

| Aspect | Detail |
|--------|--------|
| **Train** | 500 synthetic: 50 GOST + 250 copy-paste + 200 GOST-on-real-background |
| **Donors** | New set (`test_11.png`, `test_17.png`, `test_23.png`, `test_42.png`) |
| **Copy-paste backgrounds** | 20 independent images from `data/unlabeled/` |
| **GOST** | Reduced 250→50 to limit "white canvas shortcut" samples |
| **GOST-on-real-bg** | Stamp grid rendered on real unlabeled backgrounds; no synthetic doodles, realistic textures |
| **Val** | **10 real non-donor images** (`val_honest/`, regenerated to exclude new donors) |
| **Test** | **35 real non-donor, non-val images** (4 new donors + 10 new val excluded) |

**Why donors changed:** The PPI extraction pipeline (`get_ppi()` → `infer_ppi_from_dimensions()` → `estimate_paper_size_by_ratio()`) was tuned, expanding PPI coverage from 27/49 → 49/49 images. Method distribution shifted from `metadata: 23, no_ppi_available: 22, iso_inference: 2, mismatch: 2` to `iso_inference: 26, iso_override_contradiction: 14, metadata: 9`. Since `select_donors()` stratifies on PPI (2-bin quantize → 8-cell grid → rare-first), the new PPI vectors caused different donors to be selected. The algorithm is deterministic at `random_state=42` — same code produces the same donors every run.

**Why synthetic mix changed:**
- v3 had Precision=**1.0**, Recall=**0.778** — the model never false-positives on non-white backgrounds but misses stamps on messy real drawings
- Analysis: GOST images start on a blank white canvas with light gray doodles — nothing like real yellowed, creased blueprints. The model shortcuts on "white grid background = stamp".
- Fix: Reduced GOST from 250→50 (fewer easy samples), added 200 GOST-on-real-background (stamp grid on real unlabeled backgrounds) to force learning grids on realistic textures. Copy-paste kept at 250.

**Result files:** `artifacts/metrics/yolo_v4_results.json` ✅

---

### Hybrid (v4 + CV refinement)

The refiner fuses YOLOv8n confidence scores with CV features (aspect ratio, edge density)
at `cnn_weight=0.6` (config `hybrid.cnn_weight`). Improves YOLO v4 top-1 confidence F1
from **0.678 → 0.712** (+0.034).

Hybrid does **not** improve v3 (F1 unchanged at 0.857) — the v3 model already achieves
Precision=0.964 via the "white canvas shortcut"; CV refinement has nothing to add.

**Result files:** `artifacts/metrics/hybrid_results.json`

---

## Academic note

These result files are kept as a historical record of the learning process:

- `yolo_v1_results.json` — leaked (do not cite as valid)
- `yolo_v2_results.json` — partially contaminated validation (acceptable for
  hyperparameter tuning, test metrics are clean)
- `yolo_v3_results.json` — clean baseline (reference for all future work) ✅

This reflects the principle of **keeping mistakes documented rather than
erasing them**, which is stronger academic practice.
