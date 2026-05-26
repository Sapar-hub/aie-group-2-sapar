import pytest
import numpy as np
import cv2

from src.models.cv_baseline import CVBaselineDetector


class TestCVBaselinePipeline:
    def _make_stamp_image(self, width: int = 1200, height: int = 900) -> np.ndarray:
        img = np.ones((height, width, 3), dtype=np.uint8) * 255
        x, y, sw, sh = 650, 500, 400, 120

        cv2.rectangle(img, (x, y), (x + sw, y + sh), (0, 0, 0), 3)

        cv2.rectangle(img, (x + 5, y + 5), (x + sw - 5, y + sh - 5), (0, 0, 0), 1)
        cv2.rectangle(img, (x + 5, y + 30), (x + sw - 5, y + 31), (0, 0, 0), 1)
        cv2.rectangle(img, (x + 5, y + 60), (x + sw - 5, y + 61), (0, 0, 0), 1)
        cv2.rectangle(img, (x + 5, y + 90), (x + sw - 5, y + 91), (0, 0, 0), 1)

        cv2.line(img, (x + 150, y + 5), (x + 150, y + sh - 5), (0, 0, 0), 1)
        cv2.line(img, (x + 300, y + 5), (x + 300, y + sh - 5), (0, 0, 0), 1)

        cv2.putText(img, "GOST", (x + 10, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "STAMP", (x + 10, y + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "12345", (x + 10, y + 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(img, "OK", (x + 160, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)

        cv2.imwrite("/tmp/test_stamp.png", img)
        return img

    def _make_blank_image(self) -> np.ndarray:
        return np.ones((800, 600, 3), dtype=np.uint8) * 255

    def test_detect_stamp_on_synthetic(self):
        img = self._make_stamp_image()
        detector = CVBaselineDetector(min_width=200, min_height=60)
        bbox = detector.detect(img)
        assert bbox is not None, "CV baseline should detect a stamp"
        x, y, w, h = bbox
        assert 200 < w < 600
        assert 60 < h < 200
        assert 2.0 < w / h < 5.0

    def test_detect_no_stamp_on_blank(self):
        img = self._make_blank_image()
        detector = CVBaselineDetector()
        bbox = detector.detect(img)
        assert bbox is None

    def test_detect_returns_xywh(self):
        img = self._make_stamp_image()
        detector = CVBaselineDetector(min_width=200, min_height=60)
        bbox = detector.detect(img)
        assert bbox is not None
        x, y, w, h = bbox
        assert isinstance(x, int)
        assert isinstance(y, int)
        assert isinstance(w, int)
        assert isinstance(h, int)
        assert w > 0
        assert h > 0
