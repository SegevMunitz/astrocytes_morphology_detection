"""Training-target construction for nucleus-guided astrocyte instances."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from skimage.measure import regionprops

from astroseg.constants import COMPARTMENT_CLASSES


@dataclass(frozen=True)
class AstrocyteInstanceTargets:
    """Aligned targets for compartment, boundary, and nucleus-offset heads.

    Offsets point from every cell pixel toward the centroid of that cell's
    nucleus. This teaches long processes to identify their owning astrocyte.
    """

    semantic: np.ndarray
    boundary: np.ndarray
    offsets: np.ndarray
    offset_mask: np.ndarray
    instances: np.ndarray
    cell_to_nucleus: dict[int, int]


def _validate_label_image(labels: np.ndarray, name: str) -> np.ndarray:
    """Return a validated non-negative two-dimensional integer label image.

    Floating arrays are accepted only when every value is exactly integral, so
    interpolation-damaged annotations cannot silently enter instance training.
    """
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if np.any(array < 0) or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain non-negative integer labels")
    return array.astype(np.int64, copy=False)


def map_cells_to_nuclei(
    cell_instances: np.ndarray,
    nucleus_instances: np.ndarray,
    max_distance: float = 64.0,
) -> dict[int, int]:
    """Assign nearby cell and nucleus centroids one-to-one.

    GFAP masks commonly surround rather than overlap the GFAP-negative nucleus.
    Global linear assignment prevents nucleus reuse; cells without a credible
    nearby nucleus remain valid segmentation targets but receive no ownership loss.
    """
    cells = _validate_label_image(cell_instances, "cell_instances")
    nuclei = _validate_label_image(nucleus_instances, "nucleus_instances")
    if cells.shape != nuclei.shape:
        raise ValueError("Cell and nucleus instance images must have identical dimensions")
    if not np.isfinite(max_distance) or max_distance <= 0:
        raise ValueError("max_distance must be a finite positive number")
    cell_regions = regionprops(cells)
    nucleus_regions = regionprops(nuclei)
    if not cell_regions:
        raise ValueError("Cell instance annotation contains no astrocytes")
    if not nucleus_regions:
        raise ValueError("Nucleus instance image contains no nuclei")
    distances = cdist(
        np.asarray([region.centroid for region in cell_regions]),
        np.asarray([region.centroid for region in nucleus_regions]),
    )
    cell_indices, nucleus_indices = linear_sum_assignment(distances)
    mapping = {
        int(cell_regions[cell_index].label): int(nucleus_regions[nucleus_index].label)
        for cell_index, nucleus_index in zip(cell_indices, nucleus_indices, strict=True)
        if distances[cell_index, nucleus_index] <= max_distance
    }
    if not mapping:
        raise ValueError(f"No astrocyte lies within {max_distance:g} pixels of a nucleus")
    return mapping


def _cell_contact_boundaries(cells: np.ndarray, max_gap: int = 2) -> np.ndarray:
    """Mark interfaces where different cells touch or have a one-pixel gap.

    Cellpose-style labels commonly leave a watershed line of background between
    adjacent instances. Looking two pixels across that line supplies meaningful
    positive boundary supervision while ordinary cell-to-background edges remain
    excluded, preserving isolated one-pixel processes.
    """
    height, width = cells.shape
    boundary = np.zeros(cells.shape, dtype=bool)
    shifts = [
        (dy, dx)
        for dy in range(0, max_gap + 1)
        for dx in range(-max_gap, max_gap + 1)
        if (dy > 0 or dx > 0) and max(abs(dy), abs(dx)) <= max_gap
    ]
    for dy, dx in shifts:
        first_y = slice(0, height - dy) if dy >= 0 else slice(-dy, height)
        second_y = slice(dy, height) if dy >= 0 else slice(0, height + dy)
        first_x = slice(0, width - dx) if dx >= 0 else slice(-dx, width)
        second_x = slice(dx, width) if dx >= 0 else slice(0, width + dx)
        first = cells[first_y, first_x]
        second = cells[second_y, second_x]
        contact = (first > 0) & (second > 0) & (first != second)
        boundary[first_y, first_x] |= contact
        boundary[second_y, second_x] |= contact
        if max(abs(dy), abs(dx)) == 2:
            midpoint_y = slice(
                first_y.start + (1 if dy > 0 else 0),
                first_y.stop + (1 if dy > 0 else 0),
            )
            x_step = 1 if dx > 0 else (-1 if dx < 0 else 0)
            midpoint_x = slice(first_x.start + x_step, first_x.stop + x_step)
            boundary[midpoint_y, midpoint_x] |= contact
    return boundary.astype(np.uint8)


def build_astrocyte_instance_targets(
    cell_instances: np.ndarray,
    nucleus_instances: np.ndarray,
    compartments: np.ndarray | None = None,
    soma_radius: float = 20.0,
    offset_scale: float = 256.0,
    max_nucleus_distance: float = 64.0,
) -> AstrocyteInstanceTargets:
    """Build all supervision planes for the multi-head instance model.

    Explicit compartment annotations are preferred. When absent, nucleus pixels
    are class 1, cell pixels within ``soma_radius`` of their own nucleus are soma,
    and remaining cell pixels are processes; this derived target must be recorded
    as heuristic in experiments.
    """
    cells = _validate_label_image(cell_instances, "cell_instances")
    nuclei = _validate_label_image(nucleus_instances, "nucleus_instances")
    if cells.shape != nuclei.shape:
        raise ValueError("Cell and nucleus instance images must have identical dimensions")
    if not np.isfinite(soma_radius) or soma_radius <= 0:
        raise ValueError("soma_radius must be a finite positive number")
    if not np.isfinite(offset_scale) or offset_scale <= 0:
        raise ValueError("offset_scale must be a finite positive number")
    mapping = map_cells_to_nuclei(cells, nuclei, max_nucleus_distance)
    nucleus_to_cell = np.zeros(int(nuclei.max()) + 1, dtype=np.int64)
    for cell_id, nucleus_id in mapping.items():
        nucleus_to_cell[nucleus_id] = cell_id
    assigned_nucleus_cells = nucleus_to_cell[nuclei]
    assignable_nuclei = (assigned_nucleus_cells > 0) & (
        (cells == 0) | (cells == assigned_nucleus_cells)
    )
    expanded_cells = cells.copy()
    expanded_cells[assignable_nuclei] = assigned_nucleus_cells[assignable_nuclei]
    nucleus_support = np.where(assignable_nuclei, assigned_nucleus_cells, 0)

    if compartments is not None:
        semantic = _validate_label_image(compartments, "compartments")
        if semantic.shape != cells.shape:
            raise ValueError("Compartment and cell instance images must have identical dimensions")
        allowed = set(COMPARTMENT_CLASSES.values())
        if not set(np.unique(semantic)).issubset(allowed):
            raise ValueError(f"Compartment labels must be in {sorted(allowed)}")
        if np.any((semantic > 0) & (expanded_cells == 0)):
            raise ValueError("Positive compartment pixels must belong to an astrocyte instance")
        semantic = semantic.astype(np.uint8, copy=False)
    else:
        semantic = np.zeros(cells.shape, dtype=np.uint8)
        semantic[expanded_cells > 0] = COMPARTMENT_CLASSES["process"]
        distance, nearest_indices = ndi.distance_transform_edt(
            nucleus_support == 0, return_indices=True
        )
        nearest_cell = nucleus_support[tuple(nearest_indices)]
        soma = (
            (expanded_cells > 0)
            & (expanded_cells == nearest_cell)
            & (distance <= soma_radius)
        )
        semantic[soma] = COMPARTMENT_CLASSES["soma"]
        semantic[nucleus_support > 0] = COMPARTMENT_CLASSES["nucleus"]

    offsets = np.zeros((2, *cells.shape), dtype=np.float32)
    yy, xx = np.indices(cells.shape, dtype=np.float32)
    center_y = np.zeros(int(expanded_cells.max()) + 1, dtype=np.float32)
    center_x = np.zeros_like(center_y)
    for cell_id, nucleus_id in mapping.items():
        nucleus_y, nucleus_x = np.nonzero(nuclei == nucleus_id)
        center_y[cell_id] = float(nucleus_y.mean())
        center_x[cell_id] = float(nucleus_x.mean())
    mapped_cell = np.zeros(len(center_y), dtype=bool)
    mapped_cell[list(mapping)] = True
    offset_pixels = mapped_cell[expanded_cells]
    # Preserve complete displacements even when a process is longer than the
    # normalization scale. Clipping would erase the identity of long branches.
    offsets[0, offset_pixels] = (
        center_y[expanded_cells[offset_pixels]] - yy[offset_pixels]
    ) / offset_scale
    offsets[1, offset_pixels] = (
        center_x[expanded_cells[offset_pixels]] - xx[offset_pixels]
    ) / offset_scale

    boundary = _cell_contact_boundaries(expanded_cells)
    offset_mask = offset_pixels.astype(np.float32)
    return AstrocyteInstanceTargets(
        semantic=semantic,
        boundary=boundary,
        offsets=offsets,
        offset_mask=offset_mask,
        instances=expanded_cells.astype(np.uint32),
        cell_to_nucleus=mapping,
    )
