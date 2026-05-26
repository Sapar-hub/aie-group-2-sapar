#!/usr/bin/env python3
"""Train all supervised models (YOLO + Faster R-CNN) for GOST stamp detection."""

import logging
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from ultralytics import YOLO

from src.config import load_config, load_donors, load_val_honest, test_image_dir, test_label_dir, get_path
from src.data.loader import GOSTDataset
from src.evaluation.evaluate_rcnn import evaluate_rcnn
from src.evaluation.evaluate_yolo import evaluate_yolo
from src.evaluation.metrics import print_metrics
from src.models.rcnn_model import RCNNModel
from src.models.train_yolo import train_yolo

logger = logging.getLogger(__name__)


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    config_path = Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).parent.parent / "configs" / "config.yaml"

    # ─── 1. YOLO ───────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 1: Training YOLO")
    logger.info("=" * 60)

    best_pt = train_yolo(config_path)

    cfg = load_config(str(config_path.resolve()))
    donors = load_donors(cfg)
    val_honest = load_val_honest(cfg)
    exclude = donors | val_honest

    img_dir = test_image_dir(cfg)
    lbl_dir = test_label_dir(cfg)

    yolo_model = YOLO(str(best_pt))
    yolo_metrics, _ = evaluate_yolo(yolo_model, img_dir, lbl_dir, exclude)
    print_metrics(yolo_metrics, prefix="YOLO ")

    # ─── 2. Faster R-CNN ───────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Phase 2: Training Faster R-CNN")
    logger.info("=" * 60)

    rcnn_cfg = cfg.get("rcnn", {})
    rcnn = RCNNModel.from_config(cfg)
    rcnn.build()

    train_image_dir = get_path(cfg, "images_train")
    train_label_dir = get_path(cfg, "labels_train")
    synth_dpi = rcnn_cfg.get("synth_dpi", 200)

    train_dataset = GOSTDataset(train_image_dir, train_label_dir, synth_dpi=synth_dpi)
    train_loader = DataLoader(
        train_dataset,
        batch_size=rcnn_cfg.get("batch_size", 4),
        shuffle=True,
        collate_fn=lambda batch: tuple(zip(*batch)),
    )

    logger.info("Training R-CNN on %d samples (batch_size=%d, epochs=%d, lr=%s, device=%s)",
                len(train_dataset),
                rcnn_cfg.get("batch_size", 4),
                rcnn_cfg.get("num_epochs", 30),
                rcnn_cfg.get("learning_rate", 0.001),
                rcnn_cfg.get("device", "cpu"))

    rcnn.train(
        train_loader,
        num_epochs=rcnn_cfg.get("num_epochs", 30),
        lr=rcnn_cfg.get("learning_rate", 0.001),
        patience=rcnn_cfg.get("patience", 5),
    )

    artifacts_dir = get_path(cfg, "artifacts")
    model_dir = artifacts_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    weights_path = model_dir / "rcnn_best.pth"
    torch.save(rcnn.model.state_dict(), str(weights_path))
    logger.info("R-CNN weights saved to %s", weights_path)

    rcnn_metrics, _ = evaluate_rcnn(rcnn, img_dir, lbl_dir, exclude)
    print_metrics(rcnn_metrics, prefix="RCNN ")

    # ─── 3. Summary ────────────────────────────────────────────
    logger.info("=" * 60)
    logger.info("Summary")
    logger.info("=" * 60)
    print_metrics(yolo_metrics, prefix="YOLO ")
    print_metrics(rcnn_metrics, prefix="RCNN ")


if __name__ == "__main__":
    main()
