import random

import numpy as np

from .metrics import (
    bbox_iou,
    yolo_to_pixel,
    pixel_to_yolo,
    compute_metrics,
    print_metrics,
    DetectionResult,
)


def set_seeds(seed: int = 42, deterministic: bool = True):
    random.seed(seed)
    np.random.seed(seed)
    if deterministic:
        import torch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.enabled = True
        torch.use_deterministic_algorithms(True)


__all__ = [
    "bbox_iou",
    "yolo_to_pixel",
    "pixel_to_yolo",
    "best_matching_bbox",
    "compute_metrics",
    "print_metrics",
    "DetectionResult",
    "set_seeds",
]
