"""Preliminary field-level measurements from predicted masks."""

from typing import Any

import numpy as np
from skimage.measure import label

from astroseg.postprocessing.skeleton import skeleton_statistics


def extract_image_features(mask: np.ndarray) -> dict[str, Any]:
    """Measure positive area and skeleton topology for one binary image field.

    Counts are reported in pixels and connected components, without conversion
    to physical units. These are preliminary field-level, not cell-level, values.
    """
    if mask.ndim != 2 or mask.size == 0:
        raise ValueError("mask must be a non-empty 2D array")
    binary = mask.astype(bool)
    skeleton = skeleton_statistics(binary)
    return {
        "positive_pixel_count": int(binary.sum()),
        "positive_area_fraction": float(binary.mean()),
        "connected_component_count": int(label(binary, connectivity=2).max()),
        "skeleton_length": skeleton["skeleton_length"],
        "branch_point_count": skeleton["branch_point_count"],
        "endpoint_count": skeleton["endpoint_count"],
    }
