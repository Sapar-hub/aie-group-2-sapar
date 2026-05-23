"""Image quality metrics and donor selection for synthetic data generation."""

import logging
import math
from pathlib import Path
from typing import List, Tuple, Optional
from collections import defaultdict
from dataclasses import dataclass

import cv2
import numpy as np
from PIL import Image

from data.loader import load_image_and_labels

logger = logging.getLogger(__name__)


# ── PPI detection (metadata + ISO 216 inference) ──────────────────────────

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
    if width_px < 100 or height_px < 100:
        return None

    aspect_ratio_val = max(width_px, height_px) / min(width_px, height_px)
    if abs(aspect_ratio_val - ASPECT_RATIO) / ASPECT_RATIO > tolerance * 2:
        return None

    short_side_px = min(width_px, height_px)

    best_match = None
    best_score = float("inf")

    # Order by paper area ascending (A5, A4, A3, A2, A1, A0) — smaller = more common
    sorted_papers = sorted(ISO_PAPER_SIZES.items(), key=lambda x: x[1][0] * x[1][1])

    for paper_name, (w_mm, h_mm) in sorted_papers:
        paper_short_mm = min(w_mm, h_mm)
        implied_dpi = short_side_px / (paper_short_mm / 25.4)

        if not (50 < implied_dpi < 1200):
            continue

        dpi_deviation = min(abs(implied_dpi - d) for d in [72, 96, 150, 200, 300, 400, 600])
        # Score: DPI deviation + small area-based tiebreaker (prefer smaller paper)
        area_factor = (w_mm * h_mm) / (210 * 297)  # normalized to A4 area
        score = dpi_deviation + area_factor * 0.5

        if score < best_score:
            best_score = score
            best_match = paper_name

    return best_match


def infer_ppi_from_dimensions(width_px: int, height_px: int, tolerance: float = 0.1) -> Tuple[Optional[float], Optional[str]]:
    if width_px < 100 or height_px < 100:
        return None, "too_small"

    paper_size = estimate_paper_size_by_ratio(width_px, height_px, tolerance)
    if not paper_size:
        return None, "no_match"

    w_mm, h_mm = ISO_PAPER_SIZES[paper_size]

    if width_px >= height_px:
        ref_w_mm, ref_h_mm = h_mm, w_mm
    else:
        ref_w_mm, ref_h_mm = w_mm, h_mm

    ppi_w = width_px / (ref_w_mm / 25.4)
    ppi_h = height_px / (ref_h_mm / 25.4)

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


# ── Image quality & donor selection ───────────────────────────────────────


def compute_laplacian_variance(image: np.ndarray, bbox_abs: Tuple[int, int, int, int]) -> float:
    """Laplacian variance on stamp ROI as a sharpness proxy.

    Higher = sharper. Used as a noise/quality axis for stratification.

    Args:
        image: Full image (BGR).
        bbox_abs: Absolute stamp bbox (x, y, w, h).

    Returns:
        Variance of Laplacian on the stamp region.
    """
    x, y, w, h = bbox_abs
    if w <= 0 or h <= 0:
        logger.warning("Invalid bbox for laplacian: %s", bbox_abs)
        return 0.0

    roi = image[y : y + h, x : x + w]
    if roi.size == 0:
        return 0.0

    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def yolo_to_abs_bbox(label: np.ndarray, img_w: int, img_h: int) -> Tuple[int, int, int, int]:
    """Convert YOLO label [class, cx, cy, bw, bh] to (x, y, w, h)."""
    _, cx, cy, bw, bh = label
    x = int((cx - bw / 2) * img_w)
    y = int((cy - bh / 2) * img_h)
    w = int(bw * img_w)
    h = int(bh * img_h)
    return (x, y, w, h)


def _quantize_equal_width(values: List[float], n_bins: int = 2) -> List[int]:
    """Equal-width binning. Handles None by placing in bin 0."""
    bins = [0] * len(values)

    present = [(i, v) for i, v in enumerate(values) if v is not None]
    absent = [i for i, v in enumerate(values) if v is None]

    if absent:
        logger.warning(
            "Quantize: %d / %d values are None — placed in bin 0",
            len(absent), len(values),
        )

    if not present:
        return bins

    present_values = [v for _, v in present]
    sorted_vals = sorted(set(present_values))

    if len(sorted_vals) <= n_bins:
        val_to_bin = {v: i for i, v in enumerate(sorted_vals)}
        for idx, (i, v) in enumerate(present):
            bins[i] = val_to_bin[v]
        return bins

    percentiles = [
        float(np.percentile(present_values, p))
        for p in np.linspace(0, 100, n_bins + 1)[1:-1]
    ]

    for idx, (i, v) in enumerate(present):
        for bin_idx, edge in enumerate(percentiles):
            if v <= edge:
                bins[i] = bin_idx
                break
        else:
            bins[i] = n_bins - 1

    return bins


@dataclass
class _ImageFeatures:
    """Internal container for one image's stratification features."""
    idx: int
    fname: str
    ppi: Optional[float]
    laplacian: float
    roi_rel_area: float
    bin_ppi: int
    bin_lap: int
    bin_roi: int

    @property
    def cell_key(self) -> Tuple[int, int, int]:
        return (self.bin_ppi, self.bin_lap, self.bin_roi)

    @property
    def norm_vec(self) -> np.ndarray:
        n_ppi = 0.0 if self.ppi is None else self.ppi / 600.0
        n_lap = self.laplacian / 1000.0
        n_roi = self.roi_rel_area
        return np.array([n_ppi, n_lap, n_roi])


def select_donors(
    image_dir: Path,
    label_dir: Path,
    n_donors: int = 4,
    random_state: int = 42,
) -> List[str]:
    """Select donor images for synthetic data generation.

    Stratifies on PPI, Laplacian sharpness, and relative ROI area
    (2 bins each → up to 8 cells). Rare-first ordering: the least
    populated cells donate first. Within a cell the image closest to
    the cell centroid (L2 on normalised axes) is picked.

    Args:
        image_dir: Directory with original images.
        label_dir: Directory with YOLO label files.
        n_donors: Number of donors to select.
        random_state: Seed for fallback tie-breaking.

    Returns:
        Sorted list of donor filenames.

    Raises:
        ValueError: If fewer than n_donors images with valid labels exist.
    """
    rng = np.random.default_rng(random_state)

    # --- Collect features for every image ---
    image_paths = sorted(
        list(image_dir.glob("*.png")) + list(image_dir.glob("*.jpg"))
    )

    features: List[_ImageFeatures] = []
    skipped: List[Tuple[str, str]] = []

    for img_path in image_paths:
        try:
            img, labels = load_image_and_labels(img_path, label_dir)
        except (FileNotFoundError, IOError) as e:
            skipped.append((img_path.name, str(e)))
            continue

        if labels.size == 0:
            skipped.append((img_path.name, "no labels"))
            continue

        h, w = img.shape[:2]

        ppi_result = get_ppi(str(img_path))
        ppi = ppi_result.ppi

        bbox_abs = yolo_to_abs_bbox(labels[0], w, h)
        laplacian = compute_laplacian_variance(img, bbox_abs)
        roi_rel_area = (bbox_abs[2] * bbox_abs[3]) / float(w * h)

        features.append(_ImageFeatures(
            idx=len(features),
            fname=img_path.name,
            ppi=ppi,
            laplacian=laplacian,
            roi_rel_area=roi_rel_area,
            bin_ppi=0, bin_lap=0, bin_roi=0,
        ))

    if len(features) < n_donors:
        raise ValueError(
            f"Need {n_donors} donors but only {len(features)} images with labels. "
            f"Skipped: {skipped}"
        )

    logger.info("Collected features for %d / %d images", len(features), len(image_paths))
    for fname, reason in skipped:
        logger.warning("  Skipped %s: %s", fname, reason)

    # --- Quantise ---
    ppi_vals = [f.ppi for f in features]
    lap_vals = [f.laplacian for f in features]
    roi_vals = [f.roi_rel_area for f in features]

    ppi_bins = _quantize_equal_width(ppi_vals, n_bins=2)
    lap_bins = _quantize_equal_width(lap_vals, n_bins=2)
    roi_bins = _quantize_equal_width(roi_vals, n_bins=2)

    for f, bp, bl, br in zip(features, ppi_bins, lap_bins, roi_bins):
        f.bin_ppi = bp
        f.bin_lap = bl
        f.bin_roi = br

    n_ppi_none = sum(1 for v in ppi_vals if v is None)
    if n_ppi_none:
        logger.warning(
            "%d images have no PPI — stratification degraded to noise × roi only",
            n_ppi_none,
        )

    # --- Group by cell ---
    cells = defaultdict(list)
    for f in features:
        cells[f.cell_key].append(f)

    # --- Rare-first ordering ---
    sorted_cells = sorted(cells.values(), key=lambda c: len(c))

    selected: List[_ImageFeatures] = []
    selected_indices: set = set()

    for cell in sorted_cells:
        if len(selected) >= n_donors:
            break

        candidates = [f for f in cell if f.idx not in selected_indices]
        if not candidates:
            continue

        if len(candidates) == 1:
            selected.append(candidates[0])
            selected_indices.add(candidates[0].idx)
            continue

        centroid = np.mean([c.norm_vec for c in candidates], axis=0)
        best = min(candidates, key=lambda c: float(np.linalg.norm(c.norm_vec - centroid)))
        selected.append(best)
        selected_indices.add(best.idx)

    # --- Fallback: max-min diversity ---
    remaining = [f for f in features if f.idx not in selected_indices]

    while len(selected) < n_donors and remaining:
        if not selected:
            pick = rng.choice(remaining)
        else:
            pick = max(
                remaining,
                key=lambda cand: min(
                    float(np.linalg.norm(cand.norm_vec - s.norm_vec))
                    for s in selected
                ),
            )

        selected.append(pick)
        selected_indices.add(pick.idx)
        remaining = [f for f in remaining if f.idx != pick.idx]

    result = sorted(s.fname for s in selected)

    # --- Log selection table ---
    logger.info("Selected %d donor images:", len(result))
    for s in selected:
        logger.info(
            "  %-20s cell=(%d,%d,%d)  ppi=%-5s  lap=%.1f  roi=%.3f",
            s.fname, s.bin_ppi, s.bin_lap, s.bin_roi,
            f"{s.ppi:.0f}" if s.ppi is not None else "?",
            s.laplacian, s.roi_rel_area,
        )

    return result
