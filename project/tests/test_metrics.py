import pytest

from src.evaluation.metrics import (
    bbox_iou,
    yolo_to_pixel,
    pixel_to_yolo,
    compute_metrics,
    DetectionResult,
)


class TestBboxIOU:
    def test_perfect_overlap(self):
        bbox1 = (10, 10, 100, 50)
        bbox2 = (10, 10, 100, 50)
        assert bbox_iou(bbox1, bbox2) == 1.0

    def test_no_overlap(self):
        bbox1 = (0, 0, 100, 100)
        bbox2 = (200, 200, 100, 100)
        assert bbox_iou(bbox1, bbox2) == 0.0

    def test_partial_overlap(self):
        bbox1 = (0, 0, 100, 100)
        bbox2 = (50, 50, 100, 100)
        iou = bbox_iou(bbox1, bbox2)
        assert 0.1 < iou < 0.5

    def test_one_contains_other(self):
        bbox1 = (0, 0, 100, 100)
        bbox2 = (25, 25, 50, 50)
        iou = bbox_iou(bbox1, bbox2)
        assert 0.15 < iou < 0.35

    def test_zero_area_bbox(self):
        bbox1 = (0, 0, 0, 0)
        bbox2 = (10, 10, 100, 100)
        assert bbox_iou(bbox1, bbox2) == 0.0


class TestYoloConversion:
    def test_roundtrip(self):
        label = (0, 0.5, 0.5, 0.2, 0.3)
        img_w, img_h = 1000, 800
        bbox = yolo_to_pixel(label, img_w, img_h)
        yolo_back = pixel_to_yolo(bbox, img_w, img_h)
        for a, b in zip(label[1:], yolo_back):
            assert abs(a - b) < 0.01


class TestComputeMetrics:
    def test_all_found_high_iou(self):
        results = [
            DetectionResult("img1", (0, 0, 100, 50), (5, 5, 100, 50), 0.9, True),
            DetectionResult("img2", (0, 0, 100, 50), (5, 5, 100, 50), 0.85, True),
        ]
        metrics = compute_metrics(results, iou_threshold=0.5)
        assert metrics["n_images"] == 2
        assert metrics["n_found"] == 2
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0

    def test_some_missed(self):
        results = [
            DetectionResult("img1", (0, 0, 100, 50), (5, 5, 100, 50), 0.9, True),
            DetectionResult("img2", (0, 0, 100, 50), None, 0.0, False),
        ]
        metrics = compute_metrics(results, iou_threshold=0.5)
        assert metrics["n_found"] == 1
        assert metrics["detection_rate"] == 0.5

    def test_empty_results(self):
        metrics = compute_metrics([], iou_threshold=0.5)
        assert metrics == {}

    def test_precision_recall(self):
        results = [
            DetectionResult("img1", (0, 0, 100, 50), (0, 0, 100, 50), 0.9, True),
            DetectionResult("img2", (0, 0, 100, 50), (0, 0, 100, 50), 0.3, True),
            DetectionResult("img3", (0, 0, 100, 50), None, 0.0, False),
        ]
        metrics = compute_metrics(results, iou_threshold=0.5)
        assert metrics["tp"] == 1
        assert metrics["fn"] == 2
        assert metrics["recall"] == pytest.approx(0.333, rel=0.01)