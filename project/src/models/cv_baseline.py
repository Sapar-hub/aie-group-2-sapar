import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class CVBaselineDetector:
    def __init__(
        self,
        expected_ar: float = 3.55,
        ar_tolerance: float = 0.25,
        min_width: int = 100,
        min_height: int = 30,
        block_size: int = 11,
        adaptive_c: int = 5,
    ):
        self.expected_ar = expected_ar
        self.ar_tolerance = ar_tolerance
        self.min_width = min_width
        self.min_height = min_height
        self.block_size = block_size
        self.adaptive_c = adaptive_c

    def _extract_roi(self, image: np.ndarray, roi_type: str) -> Tuple[np.ndarray, Tuple[int, int, int, int]]:
        h, w = image.shape[:2]
        w_ratio, h_ratio = 0.3, 0.25
        half_w, half_h = int(w * w_ratio), int(h * h_ratio)

        if roi_type == "bottom_right":
            x, y, rw, rh = w - half_w, h - half_h, half_w, half_h
        elif roi_type == "bottom":
            x, y, rw, rh = 0, h - half_h, w, half_h
        elif roi_type == "right":
            x, y, rw, rh = w - half_w, 0, half_w, h
        else:
            x, y, rw, rh = 0, 0, w, h

        return image[y : y + rh, x : x + rw], (x, y, rw, rh)

    def _find_stamp_contours(
        self, roi: np.ndarray
    ) -> List[dict]:
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY) if len(roi.shape) == 3 else roi
        denoised = cv2.GaussianBlur(gray, (3, 3), 0)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(denoised)

        binary = cv2.adaptiveThreshold(
            enhanced,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.block_size,
            self.adaptive_c,
        )

        contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        if hierarchy is None or len(contours) == 0:
            return []

        hierarchy = hierarchy[0]
        memo = {}

        def get_depth(idx: int) -> int:
            if idx not in memo:
                parent = hierarchy[idx][3]
                memo[idx] = 0 if parent == -1 else 1 + get_depth(parent)
            return memo[idx]

        candidates = []
        for i, contour in enumerate(contours):
            if get_depth(i) == 0:
                first_child = hierarchy[i][2]
                if first_child != -1:
                    min_x, min_y = float("inf"), float("inf")
                    max_x, max_y = float("-inf"), float("-inf")
                    child_count = 0
                    j = first_child
                    while j != -1:
                        if get_depth(j) == 1:
                            cx, cy, cw, ch = cv2.boundingRect(contours[j])
                            min_x, min_y = min(min_x, cx), min(min_y, cy)
                            max_x, max_y = max(max_x, cx + cw), max(max_y, cy + ch)
                            child_count += 1
                        j = hierarchy[j][0]

                    if child_count > 1:
                        union_w, union_h = max_x - min_x, max_y - min_y
                        if union_w > self.min_width and union_h > self.min_height:
                            ar = union_w / union_h if union_h > 0 else 0
                            ratio_diff = abs(ar - self.expected_ar) / self.expected_ar
                            if ratio_diff <= self.ar_tolerance:
                                candidates.append(
                                    {
                                        "bbox": (int(min_x), int(min_y), int(union_w), int(union_h)),
                                        "ar": ar,
                                        "conf": max(0.0, 1.0 - ratio_diff / self.ar_tolerance),
                                    }
                                )

        candidates.sort(key=lambda c: (c["conf"], c["bbox"][2] * c["bbox"][3]), reverse=True)
        return candidates

    def detect(self, image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
        h, w = image.shape[:2]

        if h > w:
            roi_type = "right" if h / w > 1.3 else "bottom"
        else:
            roi_type = "bottom_right"

        roi, roi_offset = self._extract_roi(image, roi_type)
        candidates = self._find_stamp_contours(roi)

        if candidates:
            x, y, cw, ch = candidates[0]["bbox"]
            ox, oy, _, _ = roi_offset
            return (x + ox, y + oy, cw, ch)
        return None

    def detect_batch(self, images: List[np.ndarray]) -> List[Optional[Tuple[int, int, int, int]]]:
        return [self.detect(img) for img in images]


def load_config_defaults(config_path: Optional[Path] = None) -> dict:
    defaults = {
        "expected_aspect_ratio": 3.55,
        "ar_tolerance": 0.25,
        "min_width_px": 100,
        "min_height_px": 30,
        "block_size": 11,
        "adaptive_threshold_c": 5,
    }
    if config_path and config_path.exists():
        import yaml
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        cv_cfg = cfg.get("cv_baseline", {})
        defaults.update(cv_cfg)
    return defaults