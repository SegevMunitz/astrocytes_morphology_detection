"""Training-target construction for nucleus-guided astrocyte instances."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

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
) -> dict[int, int]:
    """Map every astrocyte instance to its unique overlapping nucleus.

    The nucleus with the greatest pixel overlap is selected. A cell without a
    nucleus or reuse of one nucleus by multiple cells fails because offset targets
    would otherwise encode an ambiguous biological identity.
    """
    cells = _validate_label_image(cell_instances, "cell_instances")
    nuclei = _validate_label_image(nucleus_instances, "nucleus_instances")
    if cells.shape != nuclei.shape:
        raise ValueError("Cell and nucleus instance images must have identical dimensions")
    mapping: dict[int, int] = {}
    used_nuclei: set[int] = set()
    for cell_id in np.unique(cells[cells > 0]):
        candidates = nuclei[cells == cell_id]
        candidates = candidates[candidates > 0]
        if not candidates.size:
            raise ValueError(f"Astrocyte instance {int(cell_id)} does not overlap a nucleus")
        values, counts = np.unique(candidates, return_counts=True)
        nucleus_id = int(values[np.argmax(counts)])
        if nucleus_id in used_nuclei:
            raise ValueError(f"Nucleus instance {nucleus_id} is assigned to multiple astrocytes")
        mapping[int(cell_id)] = nucleus_id
        used_nuclei.add(nucleus_id)
    if not mapping:
        raise ValueError("Cell instance annotation contains no astrocytes")
    return mapping


def _cell_contact_boundaries(cells: np.ndarray) -> np.ndarray:
    """Mark only interfaces where two different positive cell IDs touch.

    Cell-to-background edges are intentionally excluded: otherwise every pixel
    in a one-pixel-wide process would become boundary supervision and disappear.
    """
    height, width = cells.shape
    boundary = np.zeros(cells.shape, dtype=bool)
    for dy, dx in ((1, 0), (0, 1), (1, 1), (1, -1)):
        first_y = slice(0, height - dy) if dy >= 0 else slice(-dy, height)
        second_y = slice(dy, height) if dy >= 0 else slice(0, height + dy)
        first_x = slice(0, width - dx) if dx >= 0 else slice(-dx, width)
        second_x = slice(dx, width) if dx >= 0 else slice(0, width + dx)
        first = cells[first_y, first_x]
        second = cells[second_y, second_x]
        contact = (first > 0) & (second > 0) & (first != second)
        boundary[first_y, first_x] |= contact
        boundary[second_y, second_x] |= contact
    return boundary.astype(np.uint8)


def build_astrocyte_instance_targets(
    cell_instances: np.ndarray,
    nucleus_instances: np.ndarray,
    compartments: np.ndarray | None = None,
    soma_radius: float = 20.0,
    offset_scale: float = 256.0,
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
    mapping = map_cells_to_nuclei(cells, nuclei)

    if compartments is not None:
        semantic = _validate_label_image(compartments, "compartments")
        if semantic.shape != cells.shape:
            raise ValueError("Compartment and cell instance images must have identical dimensions")
        allowed = set(COMPARTMENT_CLASSES.values())
        if not set(np.unique(semantic)).issubset(allowed):
            raise ValueError(f"Compartment labels must be in {sorted(allowed)}")
        if np.any((semantic > 0) & (cells == 0)):
            raise ValueError("Positive compartment pixels must belong to an astrocyte instance")
        semantic = semantic.astype(np.uint8, copy=False)
    else:
        semantic = np.zeros(cells.shape, dtype=np.uint8)
        for cell_id, nucleus_id in mapping.items():
            cell_mask = cells == cell_id
            nucleus_mask = (nuclei == nucleus_id) & cell_mask
            distance = ndi.distance_transform_edt(~nucleus_mask)
            semantic[cell_mask] = COMPARTMENT_CLASSES["process"]
            semantic[cell_mask & (distance <= soma_radius)] = COMPARTMENT_CLASSES["soma"]
            semantic[nucleus_mask] = COMPARTMENT_CLASSES["nucleus"]

    offsets = np.zeros((2, *cells.shape), dtype=np.float32)
    yy, xx = np.indices(cells.shape, dtype=np.float32)
    for cell_id, nucleus_id in mapping.items():
        nucleus_y, nucleus_x = np.nonzero(nuclei == nucleus_id)
        center_y = float(nucleus_y.mean())
        center_x = float(nucleus_x.mean())
        cell_mask = cells == cell_id
        # Preserve complete displacements even when a process is longer than the
        # normalization scale. Clipping would erase the identity of long branches.
        offsets[0, cell_mask] = (center_y - yy[cell_mask]) / offset_scale
        offsets[1, cell_mask] = (center_x - xx[cell_mask]) / offset_scale

    boundary = _cell_contact_boundaries(cells)
    offset_mask = (cells > 0).astype(np.float32)
    return AstrocyteInstanceTargets(
        semantic=semantic,
        boundary=boundary,
        offsets=offsets,
        offset_mask=offset_mask,
        instances=cells.astype(np.uint32),
        cell_to_nucleus=mapping,
    )
