#!/usr/bin/env python3
"""Train YOLO model for GOST stamp detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ultralytics import YOLO
from evaluation.metrics import print_metrics
from evaluation.evaluate_yolo import evaluate_yolo
from models.train_yolo import train_yolo
from config import load_config, load_donors, load_val_honest, test_image_dir, test_label_dir


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    best_pt = train_yolo(config_path)

    cfg = load_config(str(config_path.resolve()))
    model = YOLO(str(best_pt))
    donors = load_donors(cfg)
    val_honest = load_val_honest(cfg)
    exclude = donors | val_honest

    img_dir = test_image_dir(cfg)
    lbl_dir = test_label_dir(cfg)
    metrics, _ = evaluate_yolo(model, img_dir, lbl_dir, exclude)
    print_metrics(metrics, prefix="YOLO ")


if __name__ == "__main__":
    main()
