"""Unified comparison of all models on the same test set (35 non-donor, non-val).

Produces a comparison table as dict and optionally saves to JSON.
"""

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np

from ..config import load_config, test_image_dir, test_label_dir, load_donors, load_val_honest
from .metrics import print_metrics

logger = logging.getLogger(__name__)


def run_comparison(
    cv_metrics: dict,
    yolo_metrics: dict,
    rcnn_metrics: Optional[dict] = None,
    hybrid_metrics: Optional[dict] = None,
    save_path: Optional[Path] = None,
) -> dict:
    rows = {
        "CV Baseline": cv_metrics,
        "YOLOv8n": yolo_metrics,
    }
    if rcnn_metrics:
        rows["Faster R-CNN"] = rcnn_metrics
    if hybrid_metrics:
        rows["Hybrid (YOLO+CV)"] = hybrid_metrics

    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'IoU':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Det.%':>6} {'n':>4}")
    print("-" * 70)
    for name, m in rows.items():
        print(
            f"{name:<22} {m.get('iou_mean', 0):>6.3f} {m.get('precision', 0):>6.3f} "
            f"{m.get('recall', 0):>6.3f} {m.get('f1', 0):>6.3f} "
            f"{m.get('detection_rate', 0) * 100:>5.1f}% {m.get('n_images', 0):>4d}"
        )
    print("=" * 70)

    if save_path:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w") as f:
            json.dump(rows, f, indent=2)
        logger.info("Comparison saved to %s", save_path)

    return rows


def print_comparison_table(rows: dict):
    print("\n" + "=" * 70)
    print(f"{'Model':<22} {'IoU':>6} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Det.%':>6} {'n':>4}")
    print("-" * 70)
    for name, m in rows.items():
        print(
            f"{name:<22} {m.get('iou_mean', 0):>6.3f} {m.get('precision', 0):>6.3f} "
            f"{m.get('recall', 0):>6.3f} {m.get('f1', 0):>6.3f} "
            f"{m.get('detection_rate', 0) * 100:>5.1f}% {m.get('n_images', 0):>4d}"
        )
    print("=" * 70)
