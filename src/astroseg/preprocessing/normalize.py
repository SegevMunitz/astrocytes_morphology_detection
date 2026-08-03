"""Intensity normalization for model inputs."""

import numpy as np


def percentile_normalize(
    image: np.ndarray,
    lower_percentile: float = 1.0,
    upper_percentile: float = 99.8,
) -> np.ndarray:
    """Scale an image to ``[0, 1]`` using percentiles.

    The result is intended for model input, not for biological intensity
    comparison across images or experiments.
    """
    if image.size == 0:
        raise ValueError("image must not be empty")
    if not np.issubdtype(image.dtype, np.number):
        raise TypeError("image must have a numeric dtype")
    if not 0.0 <= lower_percentile < upper_percentile <= 100.0:
        raise ValueError("percentiles must satisfy 0 <= lower < upper <= 100")
    if not np.isfinite(image).all():
        raise ValueError("image contains non-finite values")
    lower, upper = np.percentile(image, [lower_percentile, upper_percentile])
    if upper <= lower:
        return np.zeros(image.shape, dtype=np.float32)
    normalized = (image.astype(np.float32) - np.float32(lower)) / np.float32(upper - lower)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32, copy=False)

