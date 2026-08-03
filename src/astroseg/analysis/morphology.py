"""Connected-component morphology tables."""

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops_table


def component_morphology(mask: np.ndarray) -> pd.DataFrame:
    """Build a preliminary morphology table for binary connected components.

    Each foreground component becomes one row containing area, bounding-box
    coordinates, and eccentricity. Components are not assigned to individual cells.
    """
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    properties = regionprops_table(
        label(mask.astype(bool), connectivity=2),
        properties=("label", "area", "bbox", "eccentricity"),
    )
    return pd.DataFrame(properties)
