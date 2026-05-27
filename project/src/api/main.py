import logging
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import RedirectResponse, Response

from .schemas import HealthResponse, PredictionResponse, BoundingBox

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("gost_detector")

app = FastAPI(title="GOST Stamp Detector", version="1.0.0")

model = None
model_type = None


def _find_yolo_weights() -> Path:
    env_path = os.environ.get("MODEL_PATH")
    if env_path:
        return Path(env_path)
    return Path("artifacts/models/best.pt")


def _ensure_cuda_compatible():
    """Check if CUDA is actually usable on this GPU+PyTorch combo.
    Fall back to CPU if the GPU is incompatible with the PyTorch build."""
    try:
        result = subprocess.run(
            [sys.executable, "-c",
             "import torch; torch.zeros(1, device='cuda')"],
            capture_output=True, text=True, timeout=15,
        )
        if result.returncode != 0:
            reason = "unknown CUDA error"
            for line in result.stderr.splitlines():
                if "AcceleratorError" in line or "RuntimeError" in line:
                    reason = line.strip()
                    break
            logger.warning("CUDA detected but incompatible (%s). Falling back to CPU.", reason)
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
    except Exception as exc:
        logger.debug("CUDA compatibility check skipped: %s", exc)


def load_model():
    global model, model_type
    _ensure_cuda_compatible()

    yolo_path = _find_yolo_weights()
    rcnn_path = Path("artifacts/models/rcnn_best.pth")

    if yolo_path.exists():
        try:
            from ultralytics import YOLO
            logger.info(f"Loading YOLO model from {yolo_path}")
            model = YOLO(str(yolo_path))
            model_type = "yolo"
            logger.info("YOLO model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Failed to load YOLO model: {e}")

    if rcnn_path.exists():
        try:
            from ..models.rcnn_model import RCNNModel
            logger.info(f"Loading RCNN model from {rcnn_path}")
            model = RCNNModel()
            model.load(rcnn_path)
            model_type = "rcnn"
            logger.info("RCNN model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Failed to load RCNN model: {e}")

    logger.warning("No trained model found, using CV baseline")
    from ..models.cv_baseline import CVBaselineDetector
    model = CVBaselineDetector()
    model_type = "cv_baseline"
    logger.info("CV baseline model loaded")


@app.on_event("startup")
def startup():
    logger.info("=" * 50)
    logger.info("GOST Stamp Detection API starting up...")
    load_model()
    logger.info("=" * 50)


@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        model_loaded=model is not None,
        model_type=model_type or "none",
    )


VALID_MODELS = {"auto", "yolo", "cv"}


def _predict_yolo(model, image):
    results = model(image, verbose=False)
    result = results[0]
    if result.boxes and len(result.boxes) > 0:
        box = result.boxes[0]
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        conf = float(box.conf[0])
        return BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1), conf
    return None, None


def _predict_cv(detector, image):
    raw_bbox = detector.detect(image)
    if raw_bbox is None:
        return None, None
    x, y, bw, bh = raw_bbox
    return BoundingBox(x=x, y=y, width=bw, height=bh), 0.8


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...), model_name: str = "auto"):
    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{model_name}'. Must be one of: {', '.join(sorted(VALID_MODELS))}",
        )

    logger.info(f"Received prediction request: {file.filename}, model_name={model_name}")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        logger.info(f"Image shape: {w}x{h}")

        if model_name == "cv":
            from ..models.cv_baseline import CVBaselineDetector
            cv_detector = CVBaselineDetector()
            bbox, conf = _predict_cv(cv_detector, image)
            used_model = "cv_baseline"
        elif model_name == "yolo":
            if model_type != "yolo":
                raise HTTPException(status_code=400, detail="YOLO model not loaded at startup")
            bbox, conf = _predict_yolo(model, image)
            used_model = "yolo"
        else:
            from ..models.cv_baseline import CVBaselineDetector
            cv_detector = CVBaselineDetector()
            bbox, conf = _predict_cv(cv_detector, image)
            used_model = "cv_baseline"

            if bbox is None and model_type == "yolo":
                logger.info("CV baseline: no stamp detected, trying YOLO")
                bbox, conf = _predict_yolo(model, image)
                used_model = "yolo"

        if bbox is None:
            raise HTTPException(status_code=404, detail=f"No stamp detected (model: {used_model})")

        logger.info(f"{used_model} detection: bbox={bbox}, conf={conf:.3f}")

        return PredictionResponse(
            image_name=file.filename,
            bbox=bbox,
            confidence=conf,
            model=used_model,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/image")
async def predict_image(file: UploadFile = File(...), model_name: str = "auto"):
    if model_name not in VALID_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model_name '{model_name}'. Must be one of: {', '.join(sorted(VALID_MODELS))}",
        )

    logger.info(f"Received predict/image request: {file.filename}, model_name={model_name}")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        logger.info(f"Image shape: {w}x{h}")

        if model_name == "cv":
            from ..models.cv_baseline import CVBaselineDetector
            cv_detector = CVBaselineDetector()
            bbox, conf = _predict_cv(cv_detector, image)
            used_model = "cv_baseline"
        elif model_name == "yolo":
            if model_type != "yolo":
                raise HTTPException(status_code=400, detail="YOLO model not loaded at startup")
            bbox, conf = _predict_yolo(model, image)
            used_model = "yolo"
        else:
            from ..models.cv_baseline import CVBaselineDetector
            cv_detector = CVBaselineDetector()
            bbox, conf = _predict_cv(cv_detector, image)
            used_model = "cv_baseline"

            if bbox is None and model_type == "yolo":
                logger.info("CV baseline: no stamp detected, trying YOLO")
                bbox, conf = _predict_yolo(model, image)
                used_model = "yolo"

        if bbox is None:
            raise HTTPException(status_code=404, detail=f"No stamp detected (model: {used_model})")

        x, y, w, h = bbox.x, bbox.y, bbox.width, bbox.height
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
        cv2.putText(image, f"{used_model} {conf:.2f}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        logger.info(f"{used_model} detection: bbox={bbox}, conf={conf:.3f}")

        _, buf = cv2.imencode(".jpg", image)
        return Response(content=buf.tobytes(), media_type="image/jpeg")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Predict image error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return RedirectResponse(url="/docs")


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()