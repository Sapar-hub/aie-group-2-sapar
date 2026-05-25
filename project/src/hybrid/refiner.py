import numpy as np
import cv2
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class HybridRefiner:
    def __init__(
        self,
        expected_ar: float = 3.36,
        ar_refine_tol: float = 0.15,
        iou_threshold: float = 0.3,
        cnn_weight: float = 0.5,
    ):
        self.expected_ar = expected_ar
        self.ar_refine_tol = ar_refine_tol
        self.iou_threshold = iou_threshold
        self.cnn_weight = cnn_weight

    @classmethod
    def from_config(cls, cfg: dict):
        hybrid_cfg = cfg.get("hybrid", {})
        cv_cfg = cfg.get("cv_baseline", {})
        return cls(
            expected_ar=hybrid_cfg.get("expected_ar", 3.36),
            ar_refine_tol=hybrid_cfg.get("ar_refine_tolerance", 0.15),
            iou_threshold=hybrid_cfg.get("iou_threshold", 0.3),
            cnn_weight=hybrid_cfg.get("cnn_weight", 0.5),
        )

    def refine(
        self,
        cnn_bboxes: List[Tuple[int, int, int, int]],
        confidences: List[float],
        image: np.ndarray,
    ) -> Optional[Tuple[int, int, int, int]]:
        if not cnn_bboxes:
            return None

        candidates = []
        for (x, y, w, h), cnn_conf in zip(cnn_bboxes, confidences):
            roi = image[y : y + h, x : x + w]
            if roi.size == 0:
                continue

            ar = w / h if h > 0 else 0
            ar_diff = abs(ar - self.expected_ar) / self.expected_ar

            edges = cv2.Canny(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), 50, 150)
            edge_density = np.sum(edges > 0) / edges.size

            cv_score = max(0.0, 1.0 - ar_diff / self.ar_refine_tol)
            if edge_density > 0.1:
                cv_score = min(1.0, cv_score + 0.2)

            score = (self.cnn_weight * cnn_conf) + ((1 - self.cnn_weight) * cv_score)

            candidates.append({
                "bbox": (x, y, w, h),
                "score": score,
                "cnn_conf": cnn_conf,
                "cv_score": cv_score,
            })

        if not candidates:
            return cnn_bboxes[0] if cnn_bboxes else None

        candidates.sort(key=lambda c: c["score"], reverse=True)
        return candidates[0]["bbox"]

