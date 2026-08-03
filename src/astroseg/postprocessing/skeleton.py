"""Skeleton topology measurements."""

from typing import Any

import numpy as np
from scipy.ndimage import convolve
from skimage.morphology import skeletonize


def skeleton_statistics(mask: np.ndarray) -> dict[str, Any]:
    """Measure skeleton pixels, branch points, and endpoints in a binary field."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    skeleton = skeletonize(mask.astype(bool))
    neighbors = convolve(skeleton.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), mode="constant")
    neighbor_count = neighbors - skeleton.astype(np.uint8)
    return {
        "skeleton": skeleton,
        "skeleton_length": int(skeleton.sum()),
        "branch_point_count": int(np.logical_and(skeleton, neighbor_count >= 3).sum()),
        "endpoint_count": int(np.logical_and(skeleton, neighbor_count == 1).sum()),
    }

