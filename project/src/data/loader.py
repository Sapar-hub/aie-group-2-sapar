import cv2
import numpy as np
import torch
from pathlib import Path
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
from typing import List, Optional, Tuple


TARGET_PPI = 100
SYNTH_DPI = 200
MAX_SIZE = 800


def load_yolo_labels(label_path: Path) -> np.ndarray:
    if not label_path.exists():
        return np.empty((0, 5))
    with open(label_path, 'r') as f:
        labels = [list(map(float, line.strip().split())) for line in f]
    return np.array(labels) if labels else np.empty((0, 5))


def load_image_and_labels(image_path: Path, label_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    label_path = label_dir / f"{image_path.stem}.txt"
    if not label_path.exists():
        matching_label_files = sorted(label_dir.glob(f"*-{image_path.stem}.txt"))
        if not matching_label_files:
            raise FileNotFoundError(f"Label file for {image_path.name} not found in {label_dir}")
        label_path = matching_label_files[0]

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Image not found at {image_path}")

    labels = load_yolo_labels(label_path)
    return image, labels


def resize_ppi(
    image: np.ndarray, boxes: torch.Tensor, src_ppi: int, target_ppi: int = TARGET_PPI
) -> Tuple[np.ndarray, torch.Tensor, float]:
    scale = target_ppi / src_ppi
    if scale >= 1.0:
        return image, boxes, 1.0
    new_w = int(image.shape[1] * scale)
    new_h = int(image.shape[0] * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(boxes) > 0:
        boxes = boxes * scale
    return image, boxes, scale


def resize_maxsize(
    image: np.ndarray, boxes: torch.Tensor, max_size: int = MAX_SIZE
) -> Tuple[np.ndarray, torch.Tensor, float]:
    h, w = image.shape[:2]
    scale = max_size / max(h, w)
    if scale >= 1.0:
        return image, boxes, 1.0
    new_w, new_h = int(w * scale), int(h * scale)
    image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    if len(boxes) > 0:
        boxes = boxes * scale
    return image, boxes, scale


class GOSTDataset(Dataset):
    def __init__(self, image_dir: Path, label_dir: Path, synth_dpi: Optional[int] = None):
        self.image_dir = Path(image_dir)
        self.label_dir = Path(label_dir)
        self.synth_dpi = synth_dpi
        self.image_paths = sorted(
            list(self.image_dir.glob("*.png")) + list(self.image_dir.glob("*.jpg"))
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, dict]:
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]

        label_file = self.label_dir / f"{img_path.stem}.txt"
        if label_file.exists():
            with open(label_file) as f:
                labels = [list(map(float, line.strip().split())) for line in f if line.strip()]
        else:
            label_files = sorted(self.label_dir.glob(f"*-{img_path.stem}.txt"))
            if label_files:
                with open(label_files[0]) as f:
                    labels = [list(map(float, line.strip().split())) for line in f if line.strip()]
            else:
                labels = []

        boxes = []
        for label in labels:
            _, cx, cy, bw, bh = label
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            boxes.append([x1, y1, x2, y2])
        boxes = torch.tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)

        if self.synth_dpi is not None:
            img, boxes, _ = resize_ppi(img, boxes, self.synth_dpi)
        else:
            img, boxes, _ = resize_maxsize(img, boxes)

        img_tensor = F.to_tensor(img)
        labels_tensor = torch.ones(len(boxes), dtype=torch.int64) if len(boxes) > 0 else torch.zeros(0, dtype=torch.int64)
        target = {"boxes": boxes, "labels": labels_tensor}
        return img_tensor, target


