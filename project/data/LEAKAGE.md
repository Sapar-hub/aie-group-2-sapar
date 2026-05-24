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
| **Status** | To be computed after re-training |

**Fixes:**
1. **Copy-paste backgrounds** — already fixed since v2 (independent sources)
2. **YOLO validation** — changed from synthetic (shared donor stamps) to 10 real
   non-donor images selected via stratified sampling (PPI × sharpness × stamp size)
3. **All synthetic data for training** — removed the 80/20 split; all 500
   synthetic images used for training
4. **Test set** — reduced to 35 images (10 val + 4 donors excluded)

**Result files:** `artifacts/metrics/yolo_v3_results.json`, `yolo_v3_runs.csv`
(to be generated)

---

## Comparison table

| Version | Train | Val (YOLO) | Test | Background leak? | Val leak? | IoU | F1 |
|---------|-------|-----------|------|-----------------|-----------|-----|-----|
| v1 | 500 synth (80% split) | 92 synth | 45 real | **YES** | no | 0.880 | 1.000 |
| v2 | 408 synth (80% split) | 92 synth | 45 real | fixed | minor* | 0.765 | 0.916 |
| v3 | 500 synth (all) | 10 real | 35 real | fixed | fixed | — | — |

\* Minor: val shared donor stamps with train, but v2 metrics are on real test images so largely unaffected.

---

## Academic note

These result files are kept as a historical record of the learning process:

- `yolo_v1_results.json` — leaked (do not cite as valid)
- `yolo_v2_results.json` — partially contaminated validation (acceptable for
  hyperparameter tuning, test metrics are clean)
- `yolo_v3_results.json` — clean baseline (reference for all future work)

This reflects the principle of **keeping mistakes documented rather than
erasing them**, which is stronger academic practice.
