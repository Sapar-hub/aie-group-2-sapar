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
    crop_stamp_from_image,
    clean_background,
    resize_stamp_for_bg,
    place_by_orientation,
)
from data.image_quality import select_donors


def generate_gost_dataset(output_dir: Path, num_samples: int, dpi: int = 200, train_subdir: str = "train_v2"):
    """Generate GOST-style synthetic stamps."""
    img_dir = output_dir / "images" / train_subdir
    lbl_dir = output_dir / "labels" / train_subdir
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
    image_dir: Path,
    label_dir: Path,
    donor_fnames: set = None,
    train_subdir: str = "train_v2",
):
    """Generate copy-paste synthetic stamps.

    - Donor images: stamp region cropped, used as stamp patch
    - Non-donor images: original stamp erased with noise, used as clean background
    - Pasted stamp resized proportionally to background
    - Placement based on sheet orientation

    Args:
        donor_fnames: Set of filenames to use as stamp sources.
    """
    donor_fnames = donor_fnames or set()
    if donor_fnames:
        print(f"Using {len(donor_fnames)} donors as stamp sources: {sorted(donor_fnames)}")

    img_dir = output_dir / "images" / train_subdir
    lbl_dir = output_dir / "labels" / train_subdir
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    # Collect stamps from donors, clean backgrounds from non-donors
    donor_stamps = []
    clean_backgrounds = []

    for img_path in sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg")):
        img, labels = load_image_and_labels(img_path, label_dir)
        if labels.size == 0:
            continue

        fname = img_path.name
        if donor_fnames and fname in donor_fnames:
            stamp = crop_stamp_from_image(img, labels, margin=0.05)
            if stamp.size > 0:
                donor_stamps.append(stamp)
        else:
            cleaned = clean_background(img, labels[0])
            clean_backgrounds.append(cleaned)

    if not donor_stamps:
        print("No stamps found in donor images")
        return

    print(f"Loaded {len(donor_stamps)} stamps from {len(donor_fnames)} donors")
    print(f"Loaded {len(clean_backgrounds)} clean backgrounds")

    # Generate copy-paste samples
    for i in range(num_samples):
        if i % 50 == 0:
            print(f"Copy-paste: {i}/{num_samples}")

        bg = random.choice(clean_backgrounds)
        bh, bw = bg.shape[:2]

        stamp_patch = random.choice(donor_stamps)
        stamp_patch = resize_stamp_for_bg(stamp_patch, bw, bh)

        sh, sw = stamp_patch.shape[:2]
        x, y = place_by_orientation(bw, bh, sw, sh)

        synth = bg.copy()
        synth[y:y+sh, x:x+sw] = stamp_patch

        label = np.array([0, (x + sw / 2) / bw, (y + sh / 2) / bh, sw / bw, sh / bh])

        img_out = img_dir / f"copy_{i:04d}.png"
        cv2.imwrite(str(img_out), synth)

        lbl_out = lbl_dir / f"copy_{i:04d}.txt"
        with open(lbl_out, "w") as f:
            f.write(f"0 {label[1]:.6f} {label[2]:.6f} {label[3]:.6f} {label[4]:.6f}\n")


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
    parser.add_argument("--donor-sources", type=str, default=None,
                       help="File listing donor image filenames to use as stamp sources (one per line). "
                            "If not set, donors are auto-selected via PPI×noise×size stratification.")
    parser.add_argument("--donor-count", type=int, default=4,
                       help="Number of donor images for auto-selection (default: 4)")
    parser.add_argument("--donor-random-state", type=int, default=42,
                       help="Random state for donor selection (default: 42)")
    parser.add_argument("--train-dir", type=str, default="train_v2",
                       help="Subdirectory name under images/ and labels/ (default: train_v2)")

    args = parser.parse_args()

    output_dir = Path(args.output)
    bg_dir = Path(args.backgrounds)
    lbl_dir = Path(args.labels)

    # Determine donor set (images to use as stamp sources)
    donor_fnames = set()
    if args.donor_sources:
        donor_path = Path(args.donor_sources)
        if donor_path.exists():
            with open(donor_path) as f:
                donor_fnames = set(line.strip() for line in f if line.strip())
            print(f"Loaded {len(donor_fnames)} donors from {donor_path}")
        else:
            print(f"Donor file {donor_path} not found — skipping")
    else:
        print(f"\nAuto-selecting {args.donor_count} donors via PPI×noise×size stratification...")
        donor_fnames = set(select_donors(
            bg_dir, lbl_dir,
            n_donors=args.donor_count,
            random_state=args.donor_random_state,
        ))
        # Persist donor list for reproducibility
        output_dir.mkdir(parents=True, exist_ok=True)
        donor_path = output_dir / "donors.txt"
        with open(donor_path, "w") as f:
            for name in sorted(donor_fnames):
                f.write(f"{name}\n")
        print(f"Donor list written to {donor_path}")

    print(f"Output: {output_dir / 'images' / args.train_dir}")

    if args.num_gost > 0:
        print(f"\nGenerating {args.num_gost} GOST images...")
        generate_gost_dataset(output_dir, args.num_gost, args.dpi, args.train_dir)

    if args.num_copy > 0:
        print(f"\nGenerating {args.num_copy} copy-paste images...")
        generate_copypaste_dataset(
            output_dir, args.num_copy, bg_dir, lbl_dir,
            donor_fnames=donor_fnames, train_subdir=args.train_dir,
        )

    total = args.num_gost + args.num_copy
    print(f"\nDone! Generated {total} synthetic images in {output_dir}")


if __name__ == "__main__":
    main()