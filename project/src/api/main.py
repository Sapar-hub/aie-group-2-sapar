import logging
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, UploadFile, File, HTTPException

from api.schemas import HealthResponse, PredictionResponse, BoundingBox

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
        p = Path(env_path)
        if p.exists():
            return p

    candidates = [
        Path("artifacts/models/best.pt"),
        *sorted(Path("artifacts").glob("yolo/*/weights/best.pt")),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("artifacts/models/best.pt")


def load_model():
    global model, model_type

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
            from models.rcnn_model import RCNNModel
            logger.info(f"Loading RCNN model from {rcnn_path}")
            model = RCNNModel()
            model.load(rcnn_path)
            model_type = "rcnn"
            logger.info("RCNN model loaded successfully")
            return
        except Exception as e:
            logger.error(f"Failed to load RCNN model: {e}")

    logger.warning("No trained model found, using CV baseline")
    from models.cv_baseline import CVBaselineDetector
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


@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    logger.info(f"Received prediction request: {file.filename}")

    try:
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Could not decode image")

        h, w = image.shape[:2]
        logger.info(f"Image shape: {w}x{h}")

        if model_type == "yolo":
            results = model(image, verbose=False)
            result = results[0]
            if result.boxes and len(result.boxes) > 0:
                box = result.boxes[0]
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                bbox = BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1)
                logger.info(f"YOLO detection: bbox={bbox}, conf={conf:.3f}")
            else:
                raise HTTPException(status_code=404, detail="No stamp detected")

        elif model_type == "rcnn":
            raw_bbox = model.detect(image, conf=0.3)
            if raw_bbox is None:
                raise HTTPException(status_code=404, detail="No stamp detected")
            x, y, bw, bh = raw_bbox
            bbox = BoundingBox(x=x, y=y, width=bw, height=bh)
            conf = 0.8
            logger.info(f"RCNN detection: bbox={bbox}")

        else:
            cv_detector = model
            raw_bbox = cv_detector.detect(image)
            if raw_bbox is None:
                raise HTTPException(status_code=404, detail="No stamp detected")
            x, y, bw, bh = raw_bbox
            bbox = BoundingBox(x=x, y=y, width=bw, height=bh)
            conf = 0.8
            logger.info(f"CV detection: bbox={bbox}")

        return PredictionResponse(
            image_name=file.filename,
            bbox=bbox,
            confidence=conf,
            model=model_type,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
def root():
    return {"message": "GOST Stamp Detector API", "docs": "/docs"}


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()