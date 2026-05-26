import logging
from pathlib import Path
from typing import List, Optional, Tuple

from ultralytics import YOLO

from .metrics import (
    DetectionResult,
    bbox_iou,
    compute_metrics,
    yolo_to_pixel,
)
from . import set_seeds
from ..data.loader import load_image_and_labels

logger = logging.getLogger(__name__)


def _best_matching_bbox(
    pred_boxes, gt_bboxes: List[Tuple[int, int, int, int]]
) -> Optional[Tuple[int, int, int, int]]:
    if not pred_boxes:
        return None
    if not gt_bboxes:
        box = pred_boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        return (x1, y1, x2 - x1, y2 - y1)

    best_iou = 0.0
    best_bbox = None
    for box in pred_boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        pred_bbox = (x1, y1, x2 - x1, y2 - y1)
        for gt in gt_bboxes:
            iou = bbox_iou(pred_bbox, gt)
            if iou > best_iou:
                best_iou = iou
                best_bbox = pred_bbox
    return best_bbox


def evaluate_yolo(
    model: YOLO,
    image_dir: Path,
    label_dir: Path,
    exclude: set,
    conf_thresholds: Optional[List[float]] = None,
) -> Tuple[dict, List[DetectionResult]]:
    set_seeds()

    if conf_thresholds is None:
        conf_thresholds = [0.05, 0.1, 0.2, 0.3]

    all_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    n_eval = len(all_images) - len(exclude)
    logger.info(
        "Evaluating on %d images (%d after filtering)",
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

            preds = model(img, conf=conf, verbose=False)
            pred_boxes = preds[0].boxes if preds[0].boxes else []

            pred_bbox = _best_matching_bbox(pred_boxes, gt_bboxes)

            gt_bbox = gt_bboxes[0] if gt_bboxes else None
            iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
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
            "conf=%.2f | F1=%.3f  P=%.3f  R=%.3f  IoU=%.3f  det_rate=%.2f  (eval on %d non-excluded)",
            conf, metrics["f1"], metrics["precision"], metrics["recall"],
            metrics.get("iou_mean", 0), metrics.get("detection_rate", 0), n_eval,
        )

        if metrics["f1"] > best_f1:
            best_f1 = metrics["f1"]
            best_conf = conf
            best_metrics = metrics
            best_results = results_list

    logger.info("Best conf=%.2f (F1=%.3f)", best_conf, best_f1)
    return best_metrics, best_results
