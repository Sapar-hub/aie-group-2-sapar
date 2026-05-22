#!/usr/bin/env python3
"""CLI for generating synthetic GOST stamp datasets."""

import argparse
import random
from pathlib import Path

import cv2
import numpy as np


import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.loader import load_image_and_labels
from data.synthetic import (
    generate_synthetic_image,
    generate_synthetic_from_real,
    crop_stamp_from_image,
)


def generate_gost_dataset(output_dir: Path, num_samples: int, dpi: int = 200):
    """Generate GOST-style synthetic stamps."""
    img_dir = output_dir / "images" / "train"
    lbl_dir = output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    for i in range(num_samples):
        if i % 50 == 0:
            print(f"GOST: {i}/{num_samples}")
        img, label, _ = generate_synthetic_image(dpi=dpi)

        img_path = img_dir / f"gost_{i:04d}.png"
        cv2.imwrite(str(img_path), img)

        lbl_path = lbl_dir / f"gost_{i:04d}.txt"
        with open(lbl_path, "w") as f:
            f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def generate_copypaste_dataset(
    output_dir: Path,
    num_samples: int,
    background_dir: Path,
    label_dir: Path,
    exclude_file: Path = None,
):
    """Generate copy-paste synthetic stamps from real backgrounds.

    Args:
        exclude_file: Path to a text file listing image filenames to EXCLUDE
                      from stamp cropping and background use. One filename per line.
    """
    # Load exclude list
    exclude_fnames = set()
    if exclude_file and exclude_file.exists():
        with open(exclude_file) as f:
            exclude_fnames = set(line.strip() for line in f if line.strip())
        print(f"Excluding {len(exclude_fnames)} images from stamp sources: {sorted(exclude_fnames)}")

    img_dir = output_dir / "images" / "train"
    lbl_dir = output_dir / "labels" / "train"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Load real stamps (skip excluded)
    real_stamps = []
    for img_path in sorted(background_dir.glob("*.png")) + sorted(background_dir.glob("*.jpg")):
        if img_path.name in exclude_fnames:
            print(f"  Skipping {img_path.name} (excluded)")
            continue
        img, lbls = load_image_and_labels(img_path, label_dir)
        if lbls.size > 0:
            stamp = crop_stamp_from_image(img, lbls, margin=0.05)
            if stamp.size > 0:
                real_stamps.append((img, stamp))

    if not real_stamps:
        print("No stamps found in background_dir")
        return

    print(f"Loaded {len(real_stamps)} real stamps")

    # Load backgrounds
    backgrounds = [(img, lbls) for img, _ in real_stamps]

    for i in range(num_samples):
        if i % 50 == 0:
            print(f"Copy-paste: {i}/{num_samples}")
        bg_img, _ = random.choice(backgrounds)
        _, stamp = random.choice(real_stamps)

        synth, label, _ = generate_synthetic_from_real(bg_img, stamp)

        img_path = img_dir / f"copy_{i:04d}.png"
        cv2.imwrite(str(img_path), synth)

        lbl_path = lbl_dir / f"copy_{i:04d}.txt"
        with open(lbl_path, "w") as f:
            f.write(f"{int(label[0])} {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic GOST stamp dataset")
    parser.add_argument("--output", type=str, default="data",
                       help="Output directory (default: data)")
    parser.add_argument("--num-gost", type=int, default=250,
                       help="Number of GOST synthetic images (default: 250)")
    parser.add_argument("--num-copy", type=int, default=250,
                       help="Number of copy-paste images (default: 250)")
    parser.add_argument("--dpi", type=int, default=200,
                       help="DPI for GOST generation (default: 200)")
    parser.add_argument("--backgrounds", type=str, default="data/images/test",
                       help="Directory with background images for copy-paste")
    parser.add_argument("--labels", type=str, default="data/labels/test",
                       help="Directory with labels for copy-paste")
    parser.add_argument("--exclude-sources", type=str, default=None,
                       help="File listing image filenames to EXCLUDE from stamp sources (one per line)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    bg_dir = Path(args.backgrounds)
    lbl_dir = Path(args.labels)
    exclude_path = Path(args.exclude_sources) if args.exclude_sources else None

    print(f"Output: {output_dir / 'images' / 'train'}")

    if args.num_gost > 0:
        print(f"\nGenerating {args.num_gost} GOST images...")
        generate_gost_dataset(output_dir, args.num_gost, args.dpi)

    if args.num_copy > 0:
        print(f"\nGenerating {args.num_copy} copy-paste images...")
        generate_copypaste_dataset(output_dir, args.num_copy, bg_dir, lbl_dir, exclude_file=exclude_path)

    total = args.num_gost + args.num_copy
    print(f"\nDone! Generated {total} synthetic images in {output_dir}")


if __name__ == "__main__":
    main()