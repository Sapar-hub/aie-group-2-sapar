"""Centralized configuration loader.

Loads configs/config.yaml, resolves paths relative to project root,
and provides typed access to donors, val_honest, and all model parameters.
"""

import logging
from pathlib import Path
from typing import Optional

import yaml

logger = logging.getLogger(__name__)

_PROJECT_ROOT: Optional[Path] = None
_CACHE: Optional[dict] = None


def _find_project_root() -> Path:
    """Walk up from this file to find the project root (where configs/ lives)."""
    p = Path(__file__).resolve().parent.parent
    for _ in range(6):
        if (p / "configs").is_dir() and (p / "configs" / "config.yaml").exists():
            return p
        p = p.parent
    return Path.cwd()


def project_root() -> Path:
    global _PROJECT_ROOT
    if _PROJECT_ROOT is None:
        _PROJECT_ROOT = _find_project_root()
    return _PROJECT_ROOT


def load_config(path: Optional[str] = None) -> dict:
    global _CACHE
    if _CACHE is not None:
        return _CACHE
    config_path = Path(path) if path else project_root() / "configs" / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    _CACHE = cfg
    return cfg


def get_path(cfg: dict, key: str) -> Path:
    val = cfg.get("paths", {}).get(key)
    if val is None:
        raise KeyError(f"paths.{key} not found in config")
    p = Path(val)
    if not p.is_absolute():
        p = project_root() / p
    return p


def get_data_dir(cfg: dict) -> Path:
    d = Path(cfg.get("DATA_DIR", "data"))
    if not d.is_absolute():
        d = project_root() / d
    return d


def load_donors(cfg: Optional[dict] = None) -> set:
    if cfg is None:
        cfg = load_config()
    donors_file = get_data_dir(cfg) / "donors.txt"
    if not donors_file.exists():
        logger.warning("Donors file not found: %s", donors_file)
        return set()
    with open(donors_file) as f:
        return set(line.strip() for line in f if line.strip())


def load_val_honest(cfg: Optional[dict] = None) -> set:
    if cfg is None:
        cfg = load_config()
    val_dir = get_path(cfg, "images_val_honest")
    if not val_dir.exists():
        logger.warning("val_honest dir not found: %s", val_dir)
        return set()
    return set(p.name for p in sorted(val_dir.glob("*.png")) + sorted(val_dir.glob("*.jpg")))


def test_image_dir(cfg: Optional[dict] = None) -> Path:
    if cfg is None:
        cfg = load_config()
    return get_path(cfg, "images_test")


def test_label_dir(cfg: Optional[dict] = None) -> Path:
    if cfg is None:
        cfg = load_config()
    return get_path(cfg, "labels_test")
