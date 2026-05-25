import cv2
import numpy as np
from pathlib import Path
from typing import Tuple


def load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 5))
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]
    return np.array(labels) if labels else np.empty((0, 5))


def load_image_and_labels(image_path: Path, label_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        matching_label_files = list(label_dir.glob(f"*-{image_path.stem}.txt"))
        if not matching_label_files:
            raise FileNotFoundError(f"Label file for {image_path.name} not found in {label_dir}")
        label_path = matching_label_files[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    labels = load_yolo_labels(label_path)
    return image, labels


