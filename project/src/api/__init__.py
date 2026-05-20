import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gost_detector")


def get_model():
    from pathlib import Path
    model_path = Path("artifacts/models/best.pt")
    if model_path.exists():
        from ultralytics import YOLO
        logger.info(f"Loading model from {model_path}")
        return YOLO(str(model_path))
    else:
        logger.warning("Model not found at artifacts/models/best.pt, using cv_baseline")
        from models.cv_baseline import CVBaselineDetector
        return CVBaselineDetector()