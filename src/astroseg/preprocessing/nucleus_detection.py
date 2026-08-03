"""Classical instance detection for bright fluorescent nuclei."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.segmentation import watershed

from astroseg.preprocessing.normalize import percentile_normalize


@dataclass(frozen=True)
class NucleusDetectionResult:
    """Instance labels and diagnostics produced from one DAPI image.

    Labels use zero for background and sequential positive integers for nuclei.
    Threshold and foreground fraction make automatic batch runs auditable.
    """

    labels: np.ndarray
    binary_mask: np.ndarray
    threshold: float
    instance_count: int
    foreground_fraction: float


def _remove_small_components(binary: np.ndarray, min_area: int) -> np.ndarray:
    """Remove connected foreground regions smaller than the requested area.

    SciPy labeling keeps behavior stable across scikit-image API versions and
    returns a Boolean mask suitable for distance-transform watershed.
    """
    components, count = ndi.label(binary)
    if count == 0:
        return np.zeros(binary.shape, dtype=bool)
    areas = np.bincount(components.ravel())
    keep = areas >= min_area
    keep[0] = False
    return keep[components]


def _filter_and_relabel(labels: np.ndarray, min_area: int) -> np.ndarray:
    """Drop tiny watershed fragments and relabel retained instances sequentially.

    Sequential labels simplify storage and guarantee that the reported maximum
    equals the number of detected nuclei.
    """
    areas = np.bincount(labels.ravel())
    mapping = np.zeros(len(areas), dtype=np.uint32)
    retained = np.flatnonzero(areas >= min_area)
    retained = retained[retained != 0]
    mapping[retained] = np.arange(1, len(retained) + 1, dtype=np.uint32)
    return mapping[labels]


def detect_nucleus_instances(
    dapi: np.ndarray,
    gaussian_sigma: float = 1.2,
    threshold_scale: float = 1.0,
    min_nucleus_area: int = 30,
    min_peak_distance: int = 7,
) -> NucleusDetectionResult:
    """Detect bright nuclei using Otsu thresholding and marker watershed.

    Percentile normalization and Gaussian smoothing create a stable foreground
    mask. Distance peaks split touching nuclei without requiring a trained model.
    """
    image = np.asarray(dapi)
    if image.ndim != 2 or image.size == 0:
        raise ValueError("DAPI image must be a non-empty 2D array")
    if not np.issubdtype(image.dtype, np.number) or not np.isfinite(image).all():
        raise ValueError("DAPI image must contain finite numeric values")
    if not np.isfinite(gaussian_sigma) or gaussian_sigma < 0:
        raise ValueError("gaussian_sigma must be a finite non-negative number")
    if not np.isfinite(threshold_scale) or threshold_scale <= 0:
        raise ValueError("threshold_scale must be a finite positive number")
    if min_nucleus_area <= 0 or min_peak_distance <= 0:
        raise ValueError("Area and peak-distance parameters must be positive")

    normalized = percentile_normalize(image)
    if not np.any(normalized > 0):
        raise ValueError("DAPI channel contains no detectable intensity variation")
    smoothed = ndi.gaussian_filter(normalized, sigma=gaussian_sigma)
    otsu_threshold = float(threshold_otsu(smoothed))
    applied_threshold = float(np.clip(otsu_threshold * threshold_scale, 0.0, 1.0))
    binary = smoothed > applied_threshold
    binary = _remove_small_components(binary, min_nucleus_area)
    binary = ndi.binary_fill_holes(binary)
    binary = _remove_small_components(binary, min_nucleus_area)
    if not binary.any():
        raise ValueError("Nucleus detection produced an empty foreground mask")

    distance = ndi.distance_transform_edt(binary)
    peaks = peak_local_max(
        distance,
        labels=binary,
        min_distance=min_peak_distance,
        threshold_abs=1.0,
        exclude_border=False,
    )
    markers = np.zeros(binary.shape, dtype=np.int32)
    if len(peaks):
        markers[tuple(peaks.T)] = np.arange(1, len(peaks) + 1, dtype=np.int32)
    else:
        markers, _ = ndi.label(binary)
    labels = watershed(-distance, markers, mask=binary)
    labels = _filter_and_relabel(labels.astype(np.int64, copy=False), min_nucleus_area)
    if labels.max(initial=0) == 0:
        raise ValueError("Nucleus detection produced no instances after area filtering")
    final_binary = labels > 0
    return NucleusDetectionResult(
        labels=labels,
        binary_mask=final_binary.astype(np.float32),
        threshold=applied_threshold,
        instance_count=int(labels.max()),
        foreground_fraction=float(final_binary.mean()),
    )
