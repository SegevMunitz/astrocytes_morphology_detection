"""Simple binary-mask cleanup."""

import numpy as np
from skimage.morphology import remove_small_holes, remove_small_objects


def clean_binary_mask(
    mask: np.ndarray,
    minimum_object_size: int = 16,
    maximum_hole_size: int = 16,
) -> np.ndarray:
    """Remove small objects and fill small holes in a 2D binary mask."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    if minimum_object_size < 0 or maximum_hole_size < 0:
        raise ValueError("size thresholds must be non-negative")
    cleaned = remove_small_objects(mask.astype(bool), min_size=minimum_object_size)
    return remove_small_holes(cleaned, area_threshold=maximum_hole_size).astype(np.uint8)

