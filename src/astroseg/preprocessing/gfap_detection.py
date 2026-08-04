"""Automatic bootstrap segmentation of bright GFAP-positive structures."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

from astroseg.preprocessing.normalize import percentile_normalize


@dataclass(frozen=True)
class GfapBootstrapResult:
    """One heuristic GFAP proposal and diagnostics for later review.

    The binary mask is an automatic pseudo label, never human ground truth.
    Probabilities are uncalibrated confidence scores used for QC and selection.
    """

    mask: np.ndarray
    probabilities: np.ndarray
    low_threshold: float
    high_threshold: float
    foreground_fraction: float


def _retain_large_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Keep connected foreground components meeting the minimum pixel area.

    Component filtering removes isolated fluorescent specks while retaining long,
    thin processes whose total connected area is sufficiently large.
    """
    labels, count = ndi.label(binary)
    if count == 0:
        return np.zeros(binary.shape, dtype=bool)
    areas = np.bincount(labels.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[labels]


def _fill_small_enclosed_holes(binary: np.ndarray, max_area: int) -> np.ndarray:
    """Fill small background components that do not touch an image edge.

    Border-connected background is preserved, preventing the operation from
    turning large extracellular regions into foreground.
    """
    labels, count = ndi.label(~binary)
    if count == 0:
        return binary.copy()
    areas = np.bincount(labels.ravel())
    touches_border = np.zeros(count + 1, dtype=bool)
    touches_border[np.unique(labels[[0, -1], :])] = True
    touches_border[np.unique(labels[:, [0, -1]])] = True
    fill = (areas <= max_area) & ~touches_border
    fill[0] = False
    return binary | fill[labels]


def _hysteresis_mask(image: np.ndarray, low: float, high: float) -> np.ndarray:
    """Retain weak-signal regions connected to at least one strong pixel.

    This preserves dim GFAP processes connected to bright structures without
    accepting every low-intensity background pixel independently.
    """
    weak_labels, count = ndi.label(image >= low)
    if count == 0:
        return np.zeros(image.shape, dtype=bool)
    strong_components = np.unique(weak_labels[image >= high])
    keep = np.zeros(count + 1, dtype=bool)
    keep[strong_components] = True
    keep[0] = False
    return keep[weak_labels]


def detect_gfap_bootstrap_mask(
    gfap: np.ndarray,
    gaussian_sigma: float = 0.8,
    low_threshold_ratio: float = 0.4,
    high_threshold_scale: float = 1.0,
    min_component_area: int = 24,
    max_hole_area: int = 24,
) -> GfapBootstrapResult:
    """Create an automatic GFAP pseudo label using hysteresis thresholding.

    Otsu defines strong signal, while a lower connected threshold recovers dim
    processes. Cleanup removes tiny objects and fills only small enclosed holes.
    """
    image = np.asarray(gfap)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("GFAP image must be a non-empty 2D array")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("GFAP image must contain finite numeric values")
    if not np.isfinite(gaussian_sigma) or gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be a finite non-negative number")
    if not 0 < low_threshold_ratio < 1:
        raise ValueError("low_threshold_ratio must be between zero and one")
    if not np.isfinite(high_threshold_scale) or high_threshold_scale <= 0:
        raise ValueError("high_threshold_scale must be a finite positive number")
    if min_component_area <= 0 or max_hole_area < 0:
        raise ValueError("Component area must be positive and hole area non-negative")

    normalized = percentile_normalize(image)
    if not np.any(normalized > 0):
        raise ValueError("GFAP channel contains no detectable intensity variation")
    smoothed = ndi.gaussian_filter(normalized, sigma=gaussian_sigma)
    high = float(np.clip(threshold_otsu(smoothed) * high_threshold_scale, 0.0, 1.0))
    low = float(high * low_threshold_ratio)
    mask = _hysteresis_mask(smoothed, low, high)
    mask = _retain_large_components(mask, min_component_area)
    mask = _fill_small_enclosed_holes(mask, max_hole_area)
    if not mask.any():
        raise ValueError("GFAP bootstrap detection produced an empty pseudo mask")

    # These scores are deliberately uncalibrated. Their 0.5 decision boundary is
    # forced to match the cleaned mask while retaining relative image confidence.
    foreground = np.where(mask, 0.5 + 0.5 * smoothed, 0.5 * smoothed)
    foreground = np.clip(foreground, 0.0, 1.0).astype(np.float32, copy=False)
    probabilities = np.stack((1.0 - foreground, foreground)).astype(np.float32)
    return GfapBootstrapResult(
        mask=mask.astype(np.uint8),
        probabilities=probabilities,
        low_threshold=low,
        high_threshold=high,
        foreground_fraction=float(mask.mean()),
    )
