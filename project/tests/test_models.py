import pytest

import numpy as np
import cv2

from src.models.cv_baseline import CVBaselineDetector
from src.models.rcnn_model import RCNNModel
from src.hybrid.refiner import HybridRefiner


class TestCVBaseline:
    def test_detector_init(self):
        detector = CVBaselineDetector()
        assert detector.expected_ar == pytest.approx(3.36, abs=0.01)
        assert detector.min_width == 300


class TestRCNNModel:
    def _require_torch(self):
        pytest.importorskip("torch", reason="torch not available")

    def test_model_init(self):
        self._require_torch()
        model = RCNNModel(num_classes=2, device="cpu")
        assert model.num_classes == 2
        assert str(model.device) == "cpu"
        assert model.model is None

    def test_model_build_no_weights(self):
        self._require_torch()
        model = RCNNModel(num_classes=2, device="cpu")
        model.build(weights=None)
        assert model.model is not None
        out_features = model.model.roi_heads.box_predictor.cls_score.out_features
        assert out_features == 2

    def test_predict_before_build_raises(self):
        self._require_torch()
        model = RCNNModel()
        with pytest.raises(RuntimeError, match="not loaded"):
            model.predict(np.zeros((100, 100, 3), dtype=np.uint8))

    def test_predict_output_format(self):
        self._require_torch()
        model = RCNNModel(num_classes=2, device="cpu")
        model.build(weights=None)
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        bboxes = model.predict(img, conf=0.5)
        assert isinstance(bboxes, list)
        for bbox in bboxes:
            x, y, w, h = bbox
            assert all(isinstance(v, int) for v in (x, y, w, h))
            assert w > 0 and h > 0

    def test_detect_output(self):
        self._require_torch()
        model = RCNNModel(num_classes=2, device="cpu")
        model.build(weights=None)
        img = np.ones((400, 600, 3), dtype=np.uint8) * 255
        result = model.detect(img, conf=0.5)
        assert result is None or (len(result) == 4)


class TestHybridRefiner:
    def test_refiner_init(self):
        refiner = HybridRefiner()
        assert refiner.expected_ar == pytest.approx(3.36, abs=0.01)

    def test_refine_empty(self):
        blank = np.ones((800, 600, 3), dtype=np.uint8) * 255
        refiner = HybridRefiner()
        result = refiner.refine([], [], blank)
        assert result is None

