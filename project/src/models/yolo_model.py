from ultralytics import YOLO
from pathlib import Path
import logging
from typing import List, Optional, Tuple

logger = logging.getLogger(__name__)


class YOLOModel:
    def __init__(self, model_name: str = "yolov8n.pt", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def load(self, weights_path: Optional[Path] = None):
        if weights_path and weights_path.exists():
            logger.info(f"Loading YOLO from {weights_path}")
            self.model = YOLO(str(weights_path))
        else:
            logger.info(f"Loading pretrained YOLO: {self.model_name}")
            self.model = YOLO(self.model_name)

    def train(
        self,
        data_yaml: str,
        epochs: int = 50,
        imgsz: int = 640,
        batch: int = 16,
        project: str = "artifacts",
        name: str = "yolo_train",
        **kwargs,
    ):
        if self.model is None:
            self.model = YOLO(self.model_name)

        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            verbose=True,
            **kwargs,
        )
        return results

    def predict(self, image, conf: float = 0.25) -> List[Tuple[int, int, int, int]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        results = self.model(image, conf=conf, verbose=False)
        bboxes = []
        for r in results:
            if r.boxes and len(r.boxes) > 0:
                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    bboxes.append((x1, y1, x2 - x1, y2 - y1))
        return bboxes

    def detect(self, image, conf: float = 0.25) -> Optional[Tuple[int, int, int, int]]:
        bboxes = self.predict(image, conf)
        return bboxes[0] if bboxes else None

    def export(self, format: str = "onnx"):
        if self.model is None:
            raise RuntimeError("Model not loaded")
        return self.model.export(format=format)

    @staticmethod
    def get_best_weights(project: str, name: str) -> Path:
        return Path(project) / name / "weights" / "best.pt"