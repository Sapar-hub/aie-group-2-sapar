"""Evaluate Hybrid (YOLO + CV Refine) on test set with donor/val exclusion."""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from evaluation.metrics import (
    DetectionResult,
    bbox_iou,
    compute_metrics,
    print_metrics,
    yolo_to_pixel,
)
from hybrid.refiner import HybridRefiner
from data.loader import load_image_and_labels

logger = logging.getLogger(__name__)


def evaluate_hybrid(
    yolo_model: YOLO,
    refiner: HybridRefiner,
    image_dir: Path,
    label_dir: Path,
    exclude: set,
    conf: float = 0.1,
) -> Tuple[dict, List[DetectionResult]]:
    all_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    results = []

    for img_path in all_images:
        if img_path.name in exclude:
            continue

        img, labels = load_image_and_labels(img_path, label_dir)
        h, w = img.shape[:2]

        gt_bboxes = [yolo_to_pixel(tuple(label), w, h) for label in labels]
        gt_bbox = gt_bboxes[0] if gt_bboxes else None

        preds = yolo_model(img, conf=conf, verbose=False)
        raw_bboxes = []
        if preds[0].boxes:
            for box in preds[0].boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                raw_bboxes.append((x1, y1, x2 - x1, y2 - y1))

        pred_bbox = refiner.refine(raw_bboxes, img)

        iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else 0.0
        results.append(DetectionResult(
            image_name=img_path.name,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            iou=iou,
            found=pred_bbox is not None,
        ))

    metrics = compute_metrics(results, iou_threshold=0.5)
    logger.info("Hybrid (YOLO+CV): evaluated %d images", metrics.get("n_images", 0))
    return metrics, results


def run_hybrid_evaluation(
    yolo_weights: Path,
    config_path: Optional[str] = None,
    conf: float = 0.1,
) -> dict:
    from config import load_config, test_image_dir, test_label_dir, load_donors, load_val_honest

    cfg = load_config(config_path)
    yolo_model = YOLO(str(yolo_weights))
    refiner = HybridRefiner()

    donors = load_donors(cfg)
    val_honest = load_val_honest(cfg)
    exclude = donors | val_honest

    img_dir = test_image_dir(cfg)
    lbl_dir = test_label_dir(cfg)

    metrics, results = evaluate_hybrid(yolo_model, refiner, img_dir, lbl_dir, exclude, conf=conf)
    print_metrics(metrics, prefix="Hybrid")
    return metrics
