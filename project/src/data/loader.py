import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple


def load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 5))
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]
    return np.array(labels) if labels else np.empty((0, 5))


def load_image_and_labels(image_path: Path, label_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    matching_label_files = list(label_dir.glob(f"*-{image_path.stem}.txt"))
    if not matching_label_files:
        raise FileNotFoundError(f"Label file for {image_path.name} not found in {label_dir}")
    label_path = matching_label_files[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    labels = load_yolo_labels(label_path)
    return image, labels


def visualize_bounding_boxes(image: np.ndarray, labels: np.ndarray, class_names: List[str] = None) -> np.ndarray:
    h, w, _ = image.shape
    vis_image = image.copy()

    for label in labels:
        class_id, center_x, center_y, box_width, box_height = label
        x_min = int((center_x - box_width / 2) * w)
        y_min = int((center_y - box_height / 2) * h)
        x_max = int((center_x + box_width / 2) * w)
        y_max = int((center_y + box_height / 2) * h)
        cv2.rectangle(vis_image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        if class_names:
            text = class_names[int(class_id)]
            cv2.putText(vis_image, text, (x_min, y_min - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    return vis_image