import random

import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import cv2

logger = logging.getLogger(__name__)


class RCNNModel:
    def __init__(
        self,
        num_classes: int = 2,
        device: str = "cpu",
    ):
        self.num_classes = num_classes
        self.device = torch.device(device)
        self.model = None

    @classmethod
    def from_config(cls, cfg: dict, num_classes: int = 2):
        rcnn_cfg = cfg.get("rcnn", {})
        return cls(
            num_classes=num_classes,
            device=rcnn_cfg.get("device", "cpu"),
        )

    def build(self, weights="DEFAULT"):
        logger.info("Building Faster R-CNN model...")
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(
            in_features, self.num_classes
        )
        self.model.to(self.device)
        logger.info("Faster R-CNN model built")

    def load(self, weights_path: Path):
        if self.model is None:
            self.build()
        logger.info(f"Loading RCNN weights from {weights_path}")
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.eval()

    def evaluate(self, val_loader) -> float:
        self.model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, targets in val_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                output = self.model(images, targets)

                if isinstance(output, dict):
                    val_loss += sum(v.item() for v in output.values())
                elif isinstance(output, list):
                    for d in output:
                        if isinstance(d, dict):
                            val_loss += sum(v.item() for v in d.values())
        return val_loss / len(val_loader) if len(val_loader) > 0 else 0.0

    def set_seeds(self, seed: int = 42, deterministic: bool = True):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True)

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 30,
        lr: float = 0.001,
        patience: int = 5,
        seed: int = 42,
    ):
        self.set_seeds(seed=seed)

        if self.model is None:
            self.build()

        backbone_params = []
        head_params = []
        for name, p in self.model.named_parameters():
            if "box_predictor" in name:
                head_params.append(p)
            else:
                backbone_params.append(p)

        optimizer = torch.optim.Adam([
            {"params": backbone_params, "lr": lr * 0.1},
            {"params": head_params, "lr": lr},
        ], weight_decay=1e-4)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        best_val_loss = float("inf")
        wait = 0

        for epoch in range(num_epochs):
            self.model.train()
            loss_sum = 0
            for images, targets in train_loader:
                images = [img.to(self.device) for img in images]
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())

                if not torch.isfinite(losses):
                    logger.warning(f"NaN/Inf loss at epoch {epoch+1}, skipping batch")
                    continue

                optimizer.zero_grad()
                losses.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                loss_sum += losses.item()

            lr_scheduler.step()
            avg_loss = loss_sum / len(train_loader)

            if val_loader is not None:
                val_loss = self.evaluate(val_loader)
                logger.info(
                    f"Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_loss:.4f}, Val Loss: {val_loss:.4f}"
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logger.info(f"Early stopping at epoch {epoch+1}")
                        break
            else:
                logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

    def predict(self, image: np.ndarray, conf: float = 0.3) -> List[Tuple[int, int, int, int]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build() or load() first.")

        self.model.eval()

        h, w = image.shape[:2]
        max_size = 800
        scale = min(max_size / max(h, w), 1.0)
        if scale < 1.0:
            new_w = int(w * scale)
            new_h = int(h * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        mask = scores >= conf

        bboxes = []
        for box in boxes[mask]:
            x1, y1, x2, y2 = box
            if scale < 1.0:
                x1 /= scale
                y1 /= scale
                x2 /= scale
                y2 /= scale
            bboxes.append((int(x1), int(y1), int(x2 - x1), int(y2 - y1)))

        return bboxes

    def detect(self, image: np.ndarray, conf: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
        bboxes = self.predict(image, conf)
        if not bboxes:
            return None
        areas = [w * h for _, _, w, h in bboxes]
        return bboxes[np.argmax(areas)]