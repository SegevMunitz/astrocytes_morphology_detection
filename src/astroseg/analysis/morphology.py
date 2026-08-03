"""Connected-component morphology tables."""

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops_table


def component_morphology(mask: np.ndarray) -> pd.DataFrame:
    """Return a preliminary connected-component morphology table."""
    if mask.ndim != 2:
        raise ValueError("mask must be 2D")
    properties = regionprops_table(
        label(mask.astype(bool), connectivity=2),
        properties=("label", "area", "bbox", "eccentricity"),
    )
    return pd.DataFrame(properties)

