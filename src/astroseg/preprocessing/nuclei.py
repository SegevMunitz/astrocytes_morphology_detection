"""Validation and conversion of nucleus instance labels."""

import numpy as np


def validate_nucleus_labels(labels: np.ndarray, image_shape: tuple[int, int]) -> None:
    """Validate a nucleus instance-label array against its source image.

    Labels must be aligned, two-dimensional, finite, non-negative, and integer
    valued. The function raises on invalid data and does not modify the array.
    """
    if labels.ndim != 2:
        raise ValueError(f"Nucleus labels must be 2D; received shape {labels.shape}")
    if len(image_shape) != 2 or any(size <= 0 for size in image_shape):
        raise ValueError(f"image_shape must contain two positive dimensions; got {image_shape}")
    if labels.shape != tuple(image_shape):
        raise ValueError(f"Nucleus-label shape {labels.shape} does not match image shape {image_shape}")
    if not np.issubdtype(labels.dtype, np.number):
        raise TypeError("Nucleus labels must have a numeric dtype")
    if not np.isfinite(labels).all():
        raise ValueError("Nucleus labels contain non-finite values")
    if np.any(labels < 0):
        raise ValueError("Nucleus labels must not contain negative values")
    if np.issubdtype(labels.dtype, np.floating) and not np.equal(labels, np.floor(labels)).all():
        raise ValueError("Nucleus labels must contain integer-valued labels")


def labels_to_binary_mask(labels: np.ndarray) -> np.ndarray:
    """Convert validated nucleus instances into a binary nucleus plane.

    Every positive instance identifier becomes one and background remains zero.
    The output is float32 for direct stacking with other model input channels.
    """
    validate_nucleus_labels(labels, labels.shape if labels.ndim == 2 else (1, 1))
    return (labels > 0).astype(np.float32)
