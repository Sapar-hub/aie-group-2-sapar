from PIL import Image
from typing import Tuple, Optional
from dataclasses import dataclass
import math


ISO_PAPER_SIZES = {
    "A0": (841, 1189),
    "A1": (594, 841),
    "A2": (420, 594),
    "A3": (297, 420),
    "A4": (210, 297),
    "A5": (148, 210),
}

ASPECT_RATIO = math.sqrt(2)


@dataclass
class PPIResult:
    ppi: Optional[float]
    method: str
    paper_size: Optional[str] = None
    metadata_ppi: Optional[float] = None
    inference_ppi: Optional[float] = None


def get_metadata_ppi(img: Image.Image) -> Tuple[Optional[float], Optional[str]]:
    if "dpi" not in img.info:
        return None, "no_metadata"

    dpi_tuple = img.info["dpi"]
    if isinstance(dpi_tuple, (int, float)):
        dpi = float(dpi_tuple)
    elif isinstance(dpi_tuple, tuple) and len(dpi_tuple) >= 2:
        dpi = float(dpi_tuple[0])
    else:
        return None, "invalid_metadata"

    if 50 < dpi < 1200:
        return dpi, "metadata"
    return None, "unreasonable_dpi"


def estimate_paper_size_by_ratio(width_px: int, height_px: int, tolerance: float = 0.1) -> Optional[str]:
    width_mm, height_mm = width_px, height_px
    aspect_ratio = max(width_px, height_px) / min(width_px, height_px)
    target_ratio = ASPECT_RATIO

    if abs(aspect_ratio - target_ratio) / target_ratio > tolerance:
        return None

    for paper_name, (w_mm, h_mm) in ISO_PAPER_SIZES.items():
        if width_mm / w_mm > 0.9 and width_mm / w_mm < 1.1:
            if height_mm / h_mm > 0.9 and height_mm / h_mm < 1.1:
                return paper_name

        if width_mm / h_mm > 0.9 and width_mm / h_mm < 1.1:
            if height_mm / w_mm > 0.9 and height_mm / w_mm < 1.1:
                return paper_name

    return None


def infer_ppi_from_dimensions(width_px: int, height_px: int, tolerance: float = 0.1) -> Tuple[Optional[float], Optional[str]]:
    if width_px < 100 or height_px < 100:
        return None, "too_small"

    paper_size = estimate_paper_size_by_ratio(width_px, height_px, tolerance)
    if not paper_size:
        return None, "no_match"

    w_mm, h_mm = ISO_PAPER_SIZES[paper_size]

    ppi_w = width_px / (w_mm / 25.4)
    ppi_h = height_px / (h_mm / 25.4)

    if abs(ppi_w - ppi_h) / max(ppi_w, ppi_h) < 0.1:
        ppi = (ppi_w + ppi_h) / 2
        return ppi, "iso_inference"

    ppi = (ppi_w + ppi_h) / 2
    return ppi, "iso_inference_mismatch"


def check_ppi_contradiction(metadata_ppi: float, inference_ppi: float, tolerance: float = 0.1) -> bool:
    if abs(metadata_ppi - inference_ppi) / max(metadata_ppi, inference_ppi) <= tolerance:
        return False
    return True


def get_ppi(img_path: str) -> PPIResult:
    with Image.open(img_path) as img:
        width_px, height_px = img.size

        metadata_ppi, meta_method = get_metadata_ppi(img)
        inference_ppi, inf_method = infer_ppi_from_dimensions(width_px, height_px)

        if metadata_ppi is not None and inference_ppi is not None:
            if check_ppi_contradiction(metadata_ppi, inference_ppi):
                return PPIResult(
                    ppi=inference_ppi,
                    method="iso_override_contradiction",
                    paper_size=estimate_paper_size_by_ratio(width_px, height_px),
                    metadata_ppi=metadata_ppi,
                    inference_ppi=inference_ppi,
                )
            return PPIResult(
                ppi=metadata_ppi,
                method="metadata",
                paper_size=estimate_paper_size_by_ratio(width_px, height_px),
                metadata_ppi=metadata_ppi,
                inference_ppi=inference_ppi,
            )

        if metadata_ppi is not None:
            return PPIResult(
                ppi=metadata_ppi,
                method=meta_method,
                metadata_ppi=metadata_ppi,
            )

        if inference_ppi is not None:
            return PPIResult(
                ppi=inference_ppi,
                method=inf_method,
                paper_size=estimate_paper_size_by_ratio(width_px, height_px),
                inference_ppi=inference_ppi,
            )

        return PPIResult(ppi=None, method="no_ppi_available")


def get_dimensions(img_path: str) -> Tuple[int, int]:
    with Image.open(img_path) as img:
        return img.size