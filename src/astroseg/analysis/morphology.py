"""Connected-component morphology tables."""

import pandas as pd
import numpy as np
from skimage.measure import label, regionprops, regionprops_table

from astroseg.postprocessing.skeleton import skeleton_statistics


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


def astrocyte_instance_morphology(
    instance_labels: np.ndarray,
    compartments: np.ndarray,
    cell_to_nucleus: dict[int, int] | None = None,
) -> pd.DataFrame:
    """Measure complete cells and their nucleus/soma/process compartments.

    One row is returned per positive cell ID. Process skeleton length, branch
    points, and endpoints are computed inside that cell rather than across the
    merged microscopy field.
    """
    instances = np.asarray(instance_labels)
    classes = np.asarray(compartments)
    if instances.ndim != 2 or classes.shape != instances.shape:
        raise ValueError("Instance and compartment labels must be aligned 2D arrays")
    if np.any(instances < 0) or not np.equal(instances, np.floor(instances)).all():
        raise ValueError("Instance labels must contain non-negative integers")
    if not set(np.unique(classes)).issubset({0, 1, 2, 3}):
        raise ValueError("Compartment labels must use values 0, 1, 2, and 3")
    mapping = cell_to_nucleus or {}
    records: list[dict[str, int | float]] = []
    for region in regionprops(instances.astype(np.int32, copy=False)):
        cell_id = int(region.label)
        min_y, min_x, max_y, max_x = region.bbox
        cell_mask = region.image
        local_classes = classes[min_y:max_y, min_x:max_x]
        process_mask = cell_mask & (local_classes == 3)
        skeleton = skeleton_statistics(process_mask)
        records.append(
            {
                "cell_id": cell_id,
                "nucleus_id": int(mapping.get(cell_id, 0)),
                "area_pixels": int(region.area),
                "nucleus_area_pixels": int(np.logical_and(cell_mask, local_classes == 1).sum()),
                "soma_area_pixels": int(np.logical_and(cell_mask, local_classes == 2).sum()),
                "process_area_pixels": int(process_mask.sum()),
                "bbox_min_y": int(min_y),
                "bbox_min_x": int(min_x),
                "bbox_max_y": int(max_y),
                "bbox_max_x": int(max_x),
                "centroid_y": float(region.centroid[0]),
                "centroid_x": float(region.centroid[1]),
                "process_skeleton_length": int(skeleton["skeleton_length"]),
                "process_branch_point_count": int(skeleton["branch_point_count"]),
                "process_endpoint_count": int(skeleton["endpoint_count"]),
            }
        )
    return pd.DataFrame(records)
