#!/usr/bin/env python3
"""Generate synthetic dataset for GOST stamp detection.

Produces 3 types of synthetic images:
- GOST: grid-based stamp drawn onto a blank canvas with drawing content
- Copy-paste: real stamp cropped from donor images, pasted onto unlabeled backgrounds
- GOST-on-real-bg: grid-based stamp rendered directly onto real background images

Usage:
    python -m scripts.generate_synthetic
    python -m scripts.generate_synthetic --config path/to/config.yaml
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from data.image_quality import select_donors
from data.loader import load_image_and_labels
from data.synthetic import (
    generate_synthetic_image,
    generate_synthetic_from_real,
    generate_synthetic_image_on_background,
    crop_stamp_from_image,
)

logger = logging.getLogger("generate_synthetic")


def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def _load_config(config_path: Path) -> dict:
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    with open(config_path) as f:
        return yaml.safe_load(f)


def _save_donors(donors: list, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for d in donors:
            f.write(d + "\n")
    logger.info("Saved %d donors to %s", len(donors), path)


def _load_donor_stamps(donor_names: list, image_dir: Path, label_dir: Path) -> list:
    stamps = []
    for fname in donor_names:
        img_path = image_dir / fname
        try:
            img, labels = load_image_and_labels(img_path, label_dir)
            if labels.size == 0:
                logger.warning("No labels in donor %s, skipping", fname)
                continue
            stamp = crop_stamp_from_image(img, labels)
            if stamp.size == 0:
                logger.warning("Empty stamp crop from %s, skipping", fname)
                continue
            stamps.append(stamp)
            logger.info("Loaded donor stamp: %s (%dx%d)", fname, stamp.shape[1], stamp.shape[0])
        except Exception as e:
            logger.warning("Failed to load donor %s: %s", fname, e)
    return stamps


def _load_unlabeled_backgrounds(unlabeled_dir: Path) -> list:
    paths = sorted(unlabeled_dir.glob("*.jpg")) + sorted(unlabeled_dir.glob("*.png"))
    backgrounds = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is not None:
            backgrounds.append(img)
    logger.info("Loaded %d unlabeled backgrounds from %s", len(backgrounds), unlabeled_dir)
    return backgrounds


def _save_sample(img: np.ndarray, label: np.ndarray, img_dir: Path, lbl_dir: Path, idx: int):
    cv2.imwrite(str(img_dir / f"synth_{idx:04d}.png"), img)
    with open(lbl_dir / f"synth_{idx:04d}.txt", "w") as f:
        f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def generate_gost_batch(img_dir: Path, lbl_dir: Path, num: int, start_idx: int, dpi: int = 200):
    for i in range(num):
        idx = start_idx + i
        img, label, _ = generate_synthetic_image(dpi=dpi)
        _save_sample(img, label, img_dir, lbl_dir, idx)
    logger.info("Generated %d GOST images (idx %d..%d)", num, start_idx, start_idx + num - 1)


def generate_copypaste_batch(
    img_dir: Path, lbl_dir: Path, num: int, start_idx: int,
    donor_stamps: list, backgrounds: list,
):
    for i in range(num):
        idx = start_idx + i
        stamp = random.choice(donor_stamps)
        bg = random.choice(backgrounds)
        img, label, _ = generate_synthetic_from_real(bg, stamp)
        _save_sample(img, label, img_dir, lbl_dir, idx)
    logger.info("Generated %d copy-paste images (idx %d..%d)", num, start_idx, start_idx + num - 1)


def generate_gost_on_bg_batch(
    img_dir: Path, lbl_dir: Path, num: int, start_idx: int,
    backgrounds: list, dpi: int = 200,
):
    for i in range(num):
        idx = start_idx + i
        bg = random.choice(backgrounds)
        img, label, _ = generate_synthetic_image_on_background(bg, dpi=dpi)
        _save_sample(img, label, img_dir, lbl_dir, idx)
    logger.info("Generated %d GOST-on-bg images (idx %d..%d)", num, start_idx, start_idx + num - 1)


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GOST stamp dataset")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output image directory (labels mirror under labels/)",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--donors", type=int, default=None, help="Number of donors (overrides config)")
    args = parser.parse_args()

    setup_logging()

    config_path = Path(args.config) if args.config else Path("configs/config.yaml")
    if not config_path.exists():
        config_path = Path(__file__).resolve().parent.parent / "configs" / "config.yaml"
    cfg = _load_config(config_path)

    data_dir = Path(cfg.get("DATA_DIR", "data"))
    if not data_dir.is_absolute():
        data_dir = (config_path.parent.parent / data_dir).resolve()

    syn_cfg = cfg.get("synthetic", {})
    n_donors = args.donors or syn_cfg.get("donor_count", 4)
    random_state = syn_cfg.get("donor_random_state", 42)
    num_gost = syn_cfg.get("num_gost", 50)
    num_copypaste = syn_cfg.get("num_copypaste", 250)
    total = syn_cfg.get("num_total", 500)
    num_gost_on_bg = total - num_gost - num_copypaste

    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = data_dir / "images" / "train_v4"
    out_str = str(output_dir)
    labels_dir = Path(out_str.replace("/images/", "/labels/"))

    output_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)

    random.seed(args.seed)
    np.random.seed(args.seed)

    image_test_dir = data_dir / "images" / "test"
    label_test_dir = data_dir / "labels" / "test"
    donors_txt = data_dir / "donors.txt"

    logger.info("Selecting %d donor images from %s", n_donors, image_test_dir)
    donors = select_donors(image_test_dir, label_test_dir, n_donors=n_donors, random_state=random_state)
    _save_donors(donors, donors_txt)

    donor_stamps = _load_donor_stamps(donors, image_test_dir, label_test_dir)
    unlabeled_dir = data_dir / "unlabeled"
    backgrounds = _load_unlabeled_backgrounds(unlabeled_dir)

    offset = 0
    generate_gost_batch(output_dir, labels_dir, num_gost, offset)
    offset += num_gost

    if donor_stamps and backgrounds:
        generate_copypaste_batch(output_dir, labels_dir, num_copypaste, offset, donor_stamps, backgrounds)
    else:
        logger.warning("Skipping copy-paste: missing donor stamps or backgrounds")
    offset += num_copypaste

    if backgrounds:
        generate_gost_on_bg_batch(output_dir, labels_dir, num_gost_on_bg, offset, backgrounds)
    else:
        logger.warning("Skipping GOST-on-bg: no backgrounds available")

    generated = len(list(output_dir.glob("*.png")))
    logger.info("=" * 50)
    logger.info("Synthetic dataset generation complete!")
    logger.info("  Images: %s (%d files)", output_dir, generated)
    logger.info("  Labels: %s (%d files)", labels_dir, len(list(labels_dir.glob("*.txt"))))
    logger.info("  Donors: %s", donors_txt)


if __name__ == "__main__":
    main()
