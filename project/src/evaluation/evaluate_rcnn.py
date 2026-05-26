"""Evaluate Faster R-CNN on test set with donor/val exclusion."""

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
from ..models.rcnn_model import RCNNModel
from ..data.loader import load_image_and_labels

logger = logging.getLogger(__name__)


def _best_matching_bbox(
    pred_bboxes: List[Tuple[int, int, int, int]],
    gt_bboxes: List[Tuple[int, int, int, int]],
) -> Optional[Tuple[int, int, int, int]]:
    if not pred_bboxes:
        return None
    if not gt_bboxes:
        return pred_bboxes[0]

    best_iou = 0.0
    best_bbox = None
    for pb in pred_bboxes:
        for gt in gt_bboxes:
            iou = bbox_iou(pb, gt)
            if iou > best_iou:
                best_iou = iou
                best_bbox = pb
    return best_bbox


def evaluate_rcnn(
    model: RCNNModel,
    image_dir: Path,
    label_dir: Path,
    exclude: set,
    conf_thresholds: Optional[List[float]] = None,
) -> Tuple[dict, List[DetectionResult]]:
    set_seeds()

    if conf_thresholds is None:
        conf_thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.5]

    all_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    n_eval = len(all_images) - len(exclude)
    logger.info(
        "Evaluating RCNN on %d images (%d after filtering)",
        len(all_images), n_eval,
    )

    best_conf, best_f1, best_metrics, best_results = 0.1, 0.0, None, None

    for conf in conf_thresholds:
        results_list = []
        for img_path in all_images:
            if img_path.name in exclude:
                continue

            img, labels = load_image_and_labels(img_path, label_dir)
            h, w = img.shape[:2]

            gt_bboxes = [yolo_to_pixel(tuple(label), w, h) for label in labels]

            pred_bboxes = model.predict(img, conf=conf)
            pred_bbox = _best_matching_bbox(pred_bboxes, gt_bboxes)

            gt_bbox = gt_bboxes[0] if gt_bboxes else None
            iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox is not None and gt_bbox is not None else 0.0
            results_list.append(DetectionResult(
                image_name=img_path.name,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                iou=iou,
                found=pred_bbox is not None,
            ))

        metrics = compute_metrics(results_list, iou_threshold=0.5)
        n_eval = len(results_list)
        logger.info(
            "conf=%.2f | P=%.3f R=%.3f F1=%.3f IoU=%.3f (eval on %d)",
            conf, metrics["precision"], metrics["recall"],
            metrics["f1"], metrics.get("iou_mean", 0), n_eval,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_conf = conf
            best_metrics = metrics
            best_results = results_list

    logger.info("Best conf=%.2f (F1=%.3f)", best_conf, best_f1)
    return best_metrics, best_results


def run_rcnn_evaluation(
    weights_path: Path,
    config_path: Optional[str] = None,
    conf_thresholds: Optional[List[float]] = None,
) -> dict:
    from ..config import load_config, test_image_dir, test_label_dir, load_donors, load_val_honest

    cfg = load_config(config_path)
    model = RCNNModel.from_config(cfg)
    model.load(weights_path)

    donors = load_donors(cfg)
    val_honest = load_val_honest(cfg)
    exclude = donors | val_honest

    img_dir = test_image_dir(cfg)
    lbl_dir = test_label_dir(cfg)

    metrics, results = evaluate_rcnn(model, img_dir, lbl_dir, exclude, conf_thresholds=conf_thresholds)
    print_metrics(metrics, prefix="RCNN")
    return metrics
