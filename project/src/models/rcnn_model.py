import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.transforms import functional as F
from pathlib import Path
import logging
from typing import List, Optional, Tuple
import numpy as np

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

    def build(self):
        logger.info("Building Faster R-CNN model...")
        self.model = fasterrcnn_resnet50_fpn(weights=None)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor.cls_score = torchvision.models.detection.faster_rcnn.FastRCNNPredictor(
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

    def train(
        self,
        train_loader,
        val_loader=None,
        num_epochs: int = 30,
        lr: float = 0.001,
    ):
        if self.model is None:
            self.build()

        params = [p for p in self.model.parameters() if p.requires_grad]
        optimizer = torch.optim.SGD(params, lr=lr, momentum=0.9, weight_decay=0.0005)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        for epoch in range(num_epochs):
            self.model.train()
            loss_sum = 0
            for batch_idx, images in enumerate(train_loader):
                images = [img.to(self.device) for img in images]
                targets = [
                    {"boxes": t["boxes"].to(self.device), "labels": t["labels"].to(self.device)}
                    for t in batch_idx
                ]
                loss_dict = self.model(images, targets)
                losses = sum(loss for loss in loss_dict.values())
                optimizer.zero_grad()
                losses.backward()
                optimizer.step()
                loss_sum += losses.item()

            lr_scheduler.step()
            logger.info(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss_sum:.4f}")

            if val_loader is not None and (epoch + 1) % 5 == 0:
                metrics = self.evaluate(val_loader)
                logger.info(f"Validation: {metrics}")

    def predict(self, image: np.ndarray, conf: float = 0.3) -> List[Tuple[int, int, int, int]]:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call build() or load() first.")

        self.model.eval()
        img_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            predictions = self.model(img_tensor)[0]

        boxes = predictions["boxes"].cpu().numpy()
        scores = predictions["scores"].cpu().numpy()
        masks = scores >= conf

        bboxes = []
        for box, score in zip(boxes[masks], scores[masks]):
            x1, y1, x2, y2 = map(int, box)
            bboxes.append((x1, y1, x2 - x1, y2 - y1))

        return bboxes

    def detect(self, image: np.ndarray, conf: float = 0.3) -> Optional[Tuple[int, int, int, int]]:
        bboxes = self.predict(image, conf)
        if not bboxes:
            return None
        areas = [w * h for x, y, w, h in bboxes]
        return bboxes[np.argmax(areas)]