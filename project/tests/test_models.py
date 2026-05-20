import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import cv2


class TestCVBaseline:
    def test_detector_init(self):
        from models.cv_baseline import CVBaselineDetector
        detector = CVBaselineDetector()
        assert detector.expected_ar == 3.55
        assert detector.min_width == 100


class TestYOLOModel:
    def test_model_init(self):
        import sys
        from pathlib import Path
        from unittest.mock import MagicMock
        sys.modules['ultralytics'] = MagicMock()
        from models.yolo_model import YOLOModel
        model = YOLOModel(device="cpu")
        assert model.device == "cpu"
        assert model.model is None


class TestRCNNModel:
    @pytest.mark.skip(reason="requires torch/torchvision installed")
    def test_model_init(self):
        pass  # tested manually on Colab with GPU


class TestHybridRefiner:
    def test_refiner_init(self):
        from hybrid.refiner import HybridRefiner
        refiner = HybridRefiner()
        assert refiner.expected_ar == 3.55

    def test_refine_empty(self):
        from hybrid.refiner import HybridRefiner
        blank = np.ones((800, 600, 3), dtype=np.uint8) * 255
        refiner = HybridRefiner()
        result = refiner.refine([], blank)
        assert result is None

    def test_filter_by_ar(self):
        from hybrid.refiner import HybridRefiner
        refiner = HybridRefiner()
        bboxes = [
            (0, 0, 355, 100),
            (0, 0, 100, 100),
            (0, 0, 200, 100),
        ]
        filtered = refiner.filter_by_aspect_ratio(bboxes, ar_tol=0.25)
        assert len(filtered) == 1