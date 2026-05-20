import cv2
import numpy as np
from pathlib import Path
import random
from typing import List, Tuple, Optional
from dataclasses import dataclass

from .augment import apply_augmentations


@dataclass
class StampInfo:
    image: np.ndarray
    original_ppi: float
    original_size_px: Tuple[int, int]
    original_size_mm: Tuple[float, float]

    @property
    def aspect_ratio(self) -> float:
        w, h = self.original_size_px
        return w / h if h > 0 else 1.0


def denormalize_bbox(bbox_norm: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    class_id, center_x, center_y, width, height = bbox_norm
    x_min = int((center_x - width / 2) * img_width)
    y_min = int((center_y - height / 2) * img_height)
    x_max = int((center_x + width / 2) * img_width)
    y_max = int((center_y + height / 2) * img_height)
    return np.array([class_id, x_min, y_min, x_max, y_max])


def normalize_bbox(bbox_abs: np.ndarray, img_width: int, img_height: int) -> np.ndarray:
    class_id, x_min, y_min, x_max, y_max = bbox_abs
    center_x = ((x_min + x_max) / 2) / img_width
    center_y = ((y_min + y_max) / 2) / img_height
    width = (x_max - x_min) / img_width
    height = (y_max - y_min) / img_height
    return np.array([class_id, center_x, center_y, width, height])


def crop_stamp_regions(
    image: np.ndarray,
    labels: np.ndarray,
    ppi: Optional[float] = None,
    margin: float = 0.05
) -> List[StampInfo]:
    cropped_stamps = []
    h, w, _ = image.shape

    if ppi is None:
        for label in labels:
            _, _, _, norm_w, norm_h = label
            bbox_w_px = norm_w * w
            bbox_h_px = norm_h * h
            if bbox_w_px > 50 and bbox_h_px > 30:
                est_ppi = (bbox_w_px / 50) * 25.4
                ppi = est_ppi
                break
        if ppi is None:
            ppi = 300

    for label in labels:
        _, x_min, y_min, x_max, y_max = denormalize_bbox(label, w, h).astype(int)
        bbox_w = x_max - x_min
        bbox_h = y_max - y_min
        margin_x = int(bbox_w * margin)
        margin_y = int(bbox_h * margin)
        x_min = max(0, x_min - margin_x)
        y_min = max(0, y_min - margin_y)
        x_max = min(w, x_max + margin_x)
        y_max = min(h, y_max + margin_y)
        cropped = image[y_min:y_max, x_min:x_max]
        if cropped.size > 0:
            stamp_w, stamp_h = x_max - x_min, y_max - y_min
            cropped_stamps.append(StampInfo(
                image=cropped,
                original_ppi=ppi,
                original_size_px=(stamp_w, stamp_h),
                original_size_mm=(stamp_w / ppi * 25.4, stamp_h / ppi * 25.4)
            ))
    return cropped_stamps


def scale_stamp_to_ppi(stamp_info: StampInfo, target_ppi: float) -> np.ndarray:
    stamp_w_mm, stamp_h_mm = stamp_info.original_size_mm
    target_w = max(int(stamp_w_mm / 25.4 * target_ppi), 10)
    target_h = max(int(stamp_h_mm / 25.4 * target_ppi), 10)
    return cv2.resize(stamp_info.image, (target_w, target_h), interpolation=cv2.INTER_AREA)


def _estimate_ppi_from_size(width: int, height: int) -> float:
    if width > 9000:
        return 300
    elif width > 6000:
        return 200
    elif width > 3000:
        return 150
    else:
        return 100


def paste_stamp_onto_background(
    background_image: np.ndarray,
    stamp_patch: np.ndarray,
    target_position: Tuple[int, int]
) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
    bg_h, bg_w, _ = background_image.shape
    s_h, s_w, _ = stamp_patch.shape
    x_offset, y_offset = target_position
    x_offset = max(0, min(x_offset, bg_w - s_w))
    y_offset = max(0, min(y_offset, bg_h - s_h))
    synthetic_image = background_image.copy()
    synthetic_image[y_offset:y_offset + s_h, x_offset:x_offset + s_w] = stamp_patch
    pasted_bbox_abs = (0, x_offset, y_offset, x_offset + s_w, y_offset + s_h)
    return synthetic_image, pasted_bbox_abs


def generate_synthetic_image(
    original_images_info: List[Tuple[Path, np.ndarray, np.ndarray]],
    stamps_pool: List[StampInfo],
    num_stamps_per_image: int = 1
) -> Tuple[np.ndarray, np.ndarray]:
    if not stamps_pool:
        raise ValueError("Stamps pool cannot be empty.")
    if not original_images_info:
        raise ValueError("Original images info cannot be empty for backgrounds.")

    _, background_image, existing_labels = random.choice(original_images_info)
    bg_h, bg_w, _ = background_image.shape
    target_ppi = _estimate_ppi_from_size(bg_w, bg_h)

    all_labels = list(existing_labels) if existing_labels.size > 0 else []
    synthetic_image = background_image.copy()

    for _ in range(num_stamps_per_image):
        if not stamps_pool:
            break
        stamp_info = random.choice(stamps_pool)
        transformed_stamp = apply_augmentations(stamp_info.image)
        scaled_stamp = scale_stamp_to_ppi(StampInfo(
            image=transformed_stamp,
            original_ppi=stamp_info.original_ppi,
            original_size_px=stamp_info.original_size_px,
            original_size_mm=stamp_info.original_size_mm
        ), target_ppi)
        s_h, s_w, _ = scaled_stamp.shape
        if bg_w - s_w <= 0 or bg_h - s_h <= 0:
            continue
        x_offset = random.randint(0, bg_w - s_w)
        y_offset = random.randint(0, bg_h - s_h)
        synthetic_image, pasted_bbox_abs = paste_stamp_onto_background(
            synthetic_image, scaled_stamp, (x_offset, y_offset)
        )
        all_labels.append(normalize_bbox(np.array(pasted_bbox_abs), bg_w, bg_h))

    return synthetic_image, np.array(all_labels) if all_labels else np.empty((0, 5))