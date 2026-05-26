import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class DetectionResult:
    image_name: str
    gt_bbox: Optional[Tuple[int, int, int, int]]
    pred_bbox: Optional[Tuple[int, int, int, int]]
    iou: float
    found: bool


def bbox_iou(bbox1: Tuple[int, int, int, int], bbox2: Tuple[int, int, int, int]) -> float:
    x1, y1, w1, h1 = bbox1
    x2, y2, w2, h2 = bbox2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_w = max(0, xi2 - xi1)
    inter_h = max(0, yi2 - yi1)
    inter_area = inter_w * inter_h
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area > 0 else 0.0


def yolo_to_pixel(label: Tuple[float, ...], img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    _, cx, cy, bw, bh = label
    x = int((cx - bw / 2) * img_w)
    y = int((cy - bh / 2) * img_h)
    w = int(bw * img_w)
    h = int(bh * img_h)
    return (x, y, w, h)


def pixel_to_yolo(bbox: Tuple[int, int, int, int], img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    x, y, w, h = bbox
    cx = (x + w / 2) / img_w
    cy = (y + h / 2) / img_h
    bw = w / img_w
    bh = h / img_h
    return (cx, cy, bw, bh)


def compute_metrics(results: List[DetectionResult], iou_threshold: float = 0.5) -> dict:
    total = len(results)
    if total == 0:
        return {}

    found = [r for r in results if r.found]
    n_found = len(found)
    n_with_gt = len([r for r in results if r.gt_bbox is not None])

    ious = [r.iou for r in results]
    ious_found = [r.iou for r in found]

    tp = sum(1 for r in results if r.found and r.iou >= iou_threshold)
    fp = n_found - tp
    fn = n_with_gt - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {
        "n_images": total,
        "n_found": n_found,
        "detection_rate": n_found / total,
        "iou_mean": np.mean(ious),
        "iou_std": np.std(ious),
        "iou_median": np.median(ious),
        "iou_max": np.max(ious),
        "iou_min": np.min(ious),
        "iou_found_mean": np.mean(ious_found) if ious_found else float("nan"),
        "iou_at_threshold": sum(1 for i in ious if i >= iou_threshold) / total,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def print_metrics(metrics: dict, prefix: str = "") -> None:
    print(f"{prefix}Images:         {metrics.get('n_images', 'N/A')}")
    print(f"{prefix}Detected:        {metrics.get('n_found', 'N/A')} ({metrics.get('detection_rate', 0)*100:.1f}%)")
    print(f"{prefix}IoU mean:        {metrics.get('iou_mean', 0):.3f}")
    print(f"{prefix}IoU std:         {metrics.get('iou_std', 0):.3f}")
    print(f"{prefix}IoU median:      {metrics.get('iou_median', 0):.3f}")
    print(f"{prefix}IoU >= 0.5:      {metrics.get('iou_at_threshold', 0)*100:.1f}%")
    print(f"{prefix}Precision:       {metrics.get('precision', 0):.3f}")
    print(f"{prefix}Recall:          {metrics.get('recall', 0):.3f}")
    print(f"{prefix}F1:              {metrics.get('f1', 0):.3f}")