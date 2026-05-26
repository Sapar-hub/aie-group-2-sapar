"""Evaluate CV baseline on test set with donor/val exclusion."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np

from .metrics import (
    DetectionResult,
    bbox_iou,
    compute_metrics,
    print_metrics,
    yolo_to_pixel,
)
from . import set_seeds
from ..models.cv_baseline import CVBaselineDetector
from ..data.loader import load_image_and_labels

logger = logging.getLogger(__name__)


def evaluate_cv(
    detector: CVBaselineDetector,
    image_dir: Path,
    label_dir: Path,
    exclude: set,
) -> Tuple[dict, List[DetectionResult]]:
    set_seeds()

    all_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    results = []

    for img_path in all_images:
        if img_path.name in exclude:
            continue

        img, labels = load_image_and_labels(img_path, label_dir)
        h, w = img.shape[:2]

        gt_bboxes = [yolo_to_pixel(tuple(label), w, h) for label in labels]
        gt_bbox = gt_bboxes[0] if gt_bboxes else None

        pred_bbox = detector.detect(img)

        iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else 0.0
        results.append(DetectionResult(
            image_name=img_path.name,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            iou=iou,
            found=pred_bbox is not None,
        ))

    metrics = compute_metrics(results, iou_threshold=0.5)
    logger.info("CV Baseline: evaluated %d images", metrics.get("n_images", 0))
    return metrics, results


def run_cv_evaluation(config_path: Optional[str] = None) -> dict:
    from ..config import load_config, test_image_dir, test_label_dir, load_donors, load_val_honest

    cfg = load_config(config_path)
    detector = CVBaselineDetector.from_config(cfg)

    donors = load_donors(cfg)
    val_honest = load_val_honest(cfg)
    exclude = donors | val_honest

    img_dir = test_image_dir(cfg)
    lbl_dir = test_label_dir(cfg)

    metrics, results = evaluate_cv(detector, img_dir, lbl_dir, exclude)
    print_metrics(metrics, prefix="CV  ")
    return metrics
