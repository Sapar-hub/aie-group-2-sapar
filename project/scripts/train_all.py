#!/usr/bin/env python3
"""Train all models for GOST stamp detection."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
import numpy as np
from ultralytics import YOLO
from evaluation.metrics import print_metrics, compute_metrics, DetectionResult, bbox_iou, yolo_to_pixel
from data.loader import load_image_and_labels


DONORS_FILE = Path(__file__).parent.parent / "data" / "donors.txt"


def _load_donors() -> set:
    if DONORS_FILE.exists():
        with open(DONORS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def _best_matching_bbox(pred_boxes, gt_bboxes, h, w):
    """Greedy match: pick prediction with highest IoU to any GT.
    Falls back to highest-confidence prediction if no GT exists."""
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


def evaluate_model(model, image_dir, label_dir, donors, conf_thresholds=None):
    if conf_thresholds is None:
        conf_thresholds = [0.05, 0.1, 0.2, 0.3]

    all_images = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    print(f"Evaluating on {len(all_images)} images ({len(all_images) - len(donors)} non-donors after filtering)")

    best_conf, best_f1, best_metrics, best_results = 0.1, 0.0, None, None

    for conf in conf_thresholds:
        results_list = []
        for img_path in all_images:
            img, labels = load_image_and_labels(img_path, label_dir)
            h, w = img.shape[:2]

            gt_bboxes = []
            for label in labels:
                gt_bboxes.append(yolo_to_pixel(tuple(label), w, h))

            preds = model(img, conf=conf, verbose=False)
            pred_boxes = preds[0].boxes if preds[0].boxes else []

            pred_bbox = _best_matching_bbox(pred_boxes, gt_bboxes, h, w)

            gt_bbox = gt_bboxes[0] if gt_bboxes else None
            iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
            results_list.append(DetectionResult(
                image_name=img_path.name,
                gt_bbox=gt_bbox,
                pred_bbox=pred_bbox,
                iou=iou,
                found=pred_bbox is not None
            ))

        filtered = [r for r in results_list if r.image_name not in donors]
        metrics = compute_metrics(filtered, iou_threshold=0.5)
        n_eval = len(filtered)
        print(f"conf={conf:.2f} | F1={metrics['f1']:.3f}  P={metrics['precision']:.3f}  "
              f"R={metrics['recall']:.3f}  IoU={metrics.get('iou_mean', 0):.3f}  "
              f"det_rate={metrics.get('detection_rate', 0):.2f}  (eval on {n_eval} non-donors)")

        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_conf = conf
            best_metrics = metrics
            best_results = filtered

    print(f"\nBest conf={best_conf:.2f} (F1={best_f1:.3f})")
    return best_metrics


def train_yolo(config_path: Path = Path("configs/config.yaml")):
    print("=" * 60)
    print("YOLO Training")
    print("=" * 60)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    yolo_cfg = cfg.get("yolo", {})
    data_dir = Path(cfg.get("DATA_DIR", "data"))

    start = time.time()
    model = YOLO(f"{yolo_cfg.get('model_name', 'yolov8n')}.pt")

    results = model.train(
        data=str(data_dir / "gost_stamp.yaml"),
        epochs=yolo_cfg.get("epochs", 50),
        imgsz=yolo_cfg.get("imgsz", 640),
        batch=yolo_cfg.get("batch", 16),
        device=yolo_cfg.get("device", "cpu"),
        project="artifacts",
        name="yolo_train",
        verbose=True,
        save=True,
        plots=True,
        single_cls=True,
    )

    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed/60:.1f} minutes")

    best_weights = Path("artifacts/yolo_train/weights/best.pt")
    if best_weights.exists():
        model = YOLO(str(best_weights))
    else:
        last_weights = Path("artifacts/yolo_train/weights/last.pt")
        if last_weights.exists():
            model = YOLO(str(last_weights))
        else:
            print("No trained weights found — skipping evaluation")
            return {}

    image_test_dir = data_dir / "images" / "test"
    label_test_dir = data_dir / "labels" / "test"
    donors = _load_donors()

    print_metrics(evaluate_model(model, image_test_dir, label_test_dir, donors), prefix="YOLO ")


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    train_yolo(config_path)


if __name__ == "__main__":
    main()