"""Preliminary component-to-nucleus assignment."""

import numpy as np
from scipy.ndimage import distance_transform_edt
from skimage.measure import label


def assign_components_to_nuclei(mask: np.ndarray, nucleus_labels: np.ndarray) -> np.ndarray:
    """Assign each positive connected component to its nearest non-zero nucleus label.

    This is a spatial heuristic for later development, not a validated cell-instance
    reconstruction method.
    """
    if mask.ndim != 2 or nucleus_labels.ndim != 2 or mask.shape != nucleus_labels.shape:
        raise ValueError("mask and nucleus_labels must be aligned 2D arrays")
    if np.any(nucleus_labels < 0):
        raise ValueError("nucleus_labels must be non-negative")
    output = np.zeros(mask.shape, dtype=np.int32)
    if not np.any(nucleus_labels > 0):
        return output
    _, indices = distance_transform_edt(nucleus_labels == 0, return_indices=True)
    nearest_labels = nucleus_labels[tuple(indices)]
    components = label(mask.astype(bool))
    for component_index in range(1, int(components.max()) + 1):
        component = components == component_index
        candidate = nearest_labels[component]
        candidate = candidate[candidate > 0]
        if candidate.size:
            values, counts = np.unique(candidate, return_counts=True)
            output[component] = int(values[np.argmax(counts)])
    return output
