import logging
import shutil
from pathlib import Path
from typing import Optional

import yaml
from ultralytics import YOLO

from ..evaluation import set_seeds

logger = logging.getLogger(__name__)


def train_yolo(
    config_path: Path = Path("configs/config.yaml"),
    project: Optional[str] = None,
    name: Optional[str] = None,
    yaml_name: str = "gost_stamp.yaml",
) -> Path:
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    yolo_cfg = cfg.get("yolo", {})
    config_path = Path(config_path).resolve()
    data_dir = (config_path.parent.parent / cfg.get("DATA_DIR", "data")).resolve()

    seed = yolo_cfg.get("seed", 42)
    deterministic = yolo_cfg.get("deterministic", True)
    set_seeds(seed, deterministic)

    model_name = yolo_cfg.get("model_name", "yolov8n")
    project = project or yolo_cfg.get("project", "artifacts")
    name = name or yolo_cfg.get("name", "yolo_train")

    logger.info(
        "Training YOLO: model=%s, epochs=%s, batch=%s, device=%s, "
        "seed=%s, deterministic=%s, rect=%s, mosaic=%s",
        model_name,
        yolo_cfg.get("epochs", 50),
        yolo_cfg.get("batch", 16),
        yolo_cfg.get("device", "cpu"),
        seed,
        deterministic,
        yolo_cfg.get("rect", True),
        yolo_cfg.get("mosaic", 0.0),
    )

    model = YOLO(f"{model_name}.pt")
    output_dir = Path(project) / name
    if output_dir.exists():
        logger.warning(
            "Output directory %s already exists. Removing it to avoid YOLO auto-suffix.",
            output_dir,
        )
        shutil.rmtree(str(output_dir))
    model.train(
        data=str(data_dir / yaml_name),
        epochs=yolo_cfg.get("epochs", 50),
        imgsz=yolo_cfg.get("imgsz", 640),
        batch=yolo_cfg.get("batch", 16),
        device=yolo_cfg.get("device", "cpu"),
        seed=seed,
        deterministic=deterministic,
        rect=yolo_cfg.get("rect", True),
        mosaic=yolo_cfg.get("mosaic", 0.0),
        single_cls=yolo_cfg.get("single_cls", True),
        project=project,
        name=name,
        save=True,
        plots=True,
        verbose=True,
    )

    weights_dir = Path(project) / name / "weights"
    best_pt = weights_dir / "best.pt"
    if best_pt.exists():
        return best_pt
    last_pt = weights_dir / "last.pt"
    if last_pt.exists():
        logger.warning("best.pt not found, using last.pt")
        return last_pt
    raise FileNotFoundError(f"No trained weights found in {weights_dir}")
