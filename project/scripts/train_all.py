#!/usr/bin/env python3
"""Train all models for GOST stamp detection."""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from ultralytics import YOLO
from evaluation.metrics import print_metrics, compute_metrics, DetectionResult, bbox_iou, yolo_to_pixel
from data.loader import load_image_and_labels


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
    )

    elapsed = time.time() - start
    print(f"\nTraining time: {elapsed/60:.1f} minutes")

    best_weights = Path("artifacts/yolo_train/weights/best.pt")
    if best_weights.exists():
        model = YOLO(str(best_weights))
    else:
        model = YOLO(str(Path("artifacts/yolo_train/weights/last.pt")))

    image_test_dir = data_dir / "images" / "test"
    label_test_dir = data_dir / "labels" / "test"
    all_images = sorted(image_test_dir.glob("*.png")) + sorted(image_test_dir.glob("*.jpg"))

    results_list = []
    for img_path in all_images:
        img, labels = load_image_and_labels(img_path, label_test_dir)
        h, w = img.shape[:2]
        gt_bbox = yolo_to_pixel(tuple(labels[0]), w, h) if len(labels) > 0 else None

        preds = model(img, conf=0.1, verbose=False)
        if preds[0].boxes and len(preds[0].boxes) > 0:
            box = preds[0].boxes[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            pred_bbox = (x1, y1, x2 - x1, y2 - y1)
        else:
            pred_bbox = None

        iou = bbox_iou(pred_bbox, gt_bbox) if pred_bbox and gt_bbox else 0.0
        results_list.append(DetectionResult(
            image_name=img_path.name,
            gt_bbox=gt_bbox,
            pred_bbox=pred_bbox,
            iou=iou,
            found=pred_bbox is not None
        ))

    metrics = compute_metrics(results_list, iou_threshold=0.5)
    print_metrics(metrics, prefix="YOLO ")
    return metrics


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    train_yolo(config_path)


if __name__ == "__main__":
    main()