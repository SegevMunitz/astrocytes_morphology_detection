"""Distance-derived nucleus-proximity inputs."""

import numpy as np
from scipy.ndimage import distance_transform_edt


def create_nucleus_proximity_map(
    nucleus_mask: np.ndarray,
    max_distance: float = 64.0,
) -> np.ndarray:
    """Create a bounded proximity map to the nearest positive nucleus pixel.

    Euclidean distance is clipped at ``max_distance`` and linearly inverted, so
    nucleus pixels equal one and sufficiently distant pixels equal zero.
    """
    if nucleus_mask.ndim != 2:
        raise ValueError(f"nucleus_mask must be 2D; received shape {nucleus_mask.shape}")
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be a finite positive number")
    if not np.isfinite(nucleus_mask).all():
        raise ValueError("nucleus_mask contains non-finite values")
    if not np.isin(nucleus_mask, (0, 1)).all():
        raise ValueError("nucleus_mask must contain only binary values 0 and 1")
    binary = nucleus_mask > 0
    if not binary.any():
        return np.zeros(nucleus_mask.shape, dtype=np.float32)
    distance = distance_transform_edt(~binary)
    proximity = 1.0 - np.minimum(distance, max_distance) / max_distance
    return np.clip(proximity, 0.0, 1.0).astype(np.float32)
