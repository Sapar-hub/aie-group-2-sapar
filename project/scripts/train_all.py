#!/usr/bin/env python3
"""Train all models for GOST stamp detection."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import yaml
from ultralytics import YOLO
from evaluation.metrics import print_metrics
from evaluation.evaluate_yolo import evaluate_yolo
from models.train_yolo import train_yolo


DONORS_FILE = Path(__file__).parent.parent / "data" / "donors.txt"


def _load_donors() -> set:
    if DONORS_FILE.exists():
        with open(DONORS_FILE) as f:
            return set(line.strip() for line in f if line.strip())
    return set()


def main():
    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    best_pt = train_yolo(config_path)

    config_path = config_path.resolve()
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    data_dir = (config_path.parent.parent / cfg.get("DATA_DIR", "data")).resolve()

    model = YOLO(str(best_pt))
    donors = _load_donors()

    image_test_dir = data_dir / "images" / "test"
    label_test_dir = data_dir / "labels" / "test"
    metrics, _ = evaluate_yolo(model, image_test_dir, label_test_dir, donors)
    print_metrics(metrics, prefix="YOLO ")


if __name__ == "__main__":
    main()
