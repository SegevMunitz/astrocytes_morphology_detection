"""Nucleus-guided reconstruction of complete individual astrocytes."""

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.segmentation import watershed

from astroseg.constants import COMPARTMENT_CLASSES


@dataclass(frozen=True)
class AstrocyteInstanceResult:
    """Complete cell IDs, compartments, ownership mapping, and QC diagnostics.

    Every positive cell ID maps to exactly one detected nucleus. Compartment values
    follow background=0, nucleus=1, soma=2, process=3.
    """

    labels: np.ndarray
    compartments: np.ndarray
    nucleus_cell_labels: np.ndarray
    cell_to_nucleus: dict[int, int]
    cell_count: int
    active_nucleus_count: int
    unassigned_foreground_fraction: float
    ownership_mode: str


def _validate_probability(image: np.ndarray, name: str, shape: tuple[int, int] | None = None) -> np.ndarray:
    """Return one finite, bounded two-dimensional floating probability plane.

    An optional expected shape enforces exact registration with nucleus labels.
    """
    array = np.asarray(image)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if shape is not None and array.shape != shape:
        raise ValueError(f"{name} shape {array.shape} does not match {shape}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array.astype(np.float32, copy=False)


def _active_nucleus_markers(
    nucleus_labels: np.ndarray,
    foreground: np.ndarray,
    max_distance: float,
) -> tuple[np.ndarray, dict[int, int]]:
    """Create sequential markers for nuclei supported by nearby GFAP foreground.

    Filtering prevents every DAPI-positive non-astrocyte nucleus from becoming an
    astrocyte instance when mixed cell types are present.
    """
    distance_to_foreground = ndi.distance_transform_edt(~foreground)
    markers = np.zeros(foreground.shape, dtype=np.int32)
    cell_to_nucleus: dict[int, int] = {}
    next_cell = 1
    for nucleus_id in np.unique(nucleus_labels[nucleus_labels > 0]):
        nucleus_mask = nucleus_labels == nucleus_id
        if float(distance_to_foreground[nucleus_mask].min()) <= max_distance:
            markers[nucleus_mask] = next_cell
            cell_to_nucleus[next_cell] = int(nucleus_id)
            next_cell += 1
    return markers, cell_to_nucleus


def separate_astrocyte_instances(
    cell_probability: np.ndarray,
    nucleus_labels: np.ndarray,
    semantic_probabilities: np.ndarray | None = None,
    boundary_probability: np.ndarray | None = None,
    ownership_offsets: np.ndarray | None = None,
    foreground_threshold: float = 0.5,
    boundary_threshold: float = 0.5,
    max_nucleus_to_gfap_distance: float = 16.0,
    offset_scale: float = 256.0,
    max_offset_endpoint_distance: float = 32.0,
    soma_expansion: int = 4,
    soma_radius: float = 20.0,
    min_cell_area: int = 50,
    min_gfap_area: int = 20,
) -> AstrocyteInstanceResult:
    """Assign soma and processes to nuclei and return individual cell instances.

    Learned ownership offsets are the primary assignment mechanism. Each cell
    pixel votes for the nucleus reached by its predicted vector. Boundary-aware
    watershed fills unreliable votes and provides the bootstrap path before a
    trained instance checkpoint exists.
    """
    probability = _validate_probability(cell_probability, "cell_probability")
    nuclei = np.asarray(nucleus_labels)
    if nuclei.ndim != 2 or nuclei.shape != probability.shape:
        raise ValueError("nucleus_labels must be a 2D array aligned with cell_probability")
    if not np.issubdtype(nuclei.dtype, np.number) or not np.isfinite(nuclei).all():
        raise ValueError("nucleus_labels must contain finite numeric values")
    if np.any(nuclei < 0) or not np.equal(nuclei, np.floor(nuclei)).all():
        raise ValueError("nucleus_labels must contain non-negative integers")
    for name, value in (
        ("foreground_threshold", foreground_threshold),
        ("boundary_threshold", boundary_threshold),
    ):
        if not 0 < value < 1:
            raise ValueError(f"{name} must be between zero and one")
    positive_values = (
        max_nucleus_to_gfap_distance,
        offset_scale,
        max_offset_endpoint_distance,
        soma_radius,
    )
    if any(not np.isfinite(value) or value <= 0 for value in positive_values):
        raise ValueError("Distance and offset parameters must be finite and positive")
    if soma_expansion < 0 or min_cell_area <= 0 or min_gfap_area <= 0:
        raise ValueError("Expansion must be non-negative and area thresholds positive")

    foreground = probability >= foreground_threshold
    if not foreground.any():
        raise ValueError("Cell probability produced an empty astrocyte foreground")
    markers, preliminary_mapping = _active_nucleus_markers(
        nuclei, foreground, max_nucleus_to_gfap_distance
    )
    if not preliminary_mapping:
        raise ValueError("No detected nucleus is sufficiently close to GFAP foreground")

    if boundary_probability is None:
        boundary = np.zeros(probability.shape, dtype=np.float32)
    else:
        boundary = _validate_probability(
            boundary_probability, "boundary_probability", probability.shape
        )
    elevation = (1.0 - probability) + boundary
    watershed_partition = watershed(elevation, markers=markers, watershed_line=True)

    ownership_mode = "watershed_bootstrap"
    assignment = watershed_partition.astype(np.int32, copy=False)
    if ownership_offsets is not None:
        offsets = np.asarray(ownership_offsets)
        if offsets.shape != (2, *probability.shape):
            raise ValueError("ownership_offsets must have shape [2, H, W]")
        if not np.issubdtype(offsets.dtype, np.number) or not np.isfinite(offsets).all():
            raise ValueError("ownership_offsets must contain finite numeric values")
        yy, xx = np.indices(probability.shape, dtype=np.float32)
        # Do not blur ownership across thin neighboring processes: their vectors
        # may intentionally point toward different nuclei only one pixel apart.
        endpoint_y = np.rint(yy + offsets[0] * offset_scale).astype(np.int64)
        endpoint_x = np.rint(xx + offsets[1] * offset_scale).astype(np.int64)
        endpoint_y = np.clip(endpoint_y, 0, probability.shape[0] - 1)
        endpoint_x = np.clip(endpoint_x, 0, probability.shape[1] - 1)
        endpoint_distance, nearest_indices = ndi.distance_transform_edt(
            markers == 0, return_indices=True
        )
        nearest_marker = markers[tuple(nearest_indices)]
        voted_marker = nearest_marker[endpoint_y, endpoint_x]
        reliable = endpoint_distance[endpoint_y, endpoint_x] <= max_offset_endpoint_distance
        assignment = np.where(reliable, voted_marker, watershed_partition).astype(np.int32)
        ownership_mode = "learned_offsets_with_watershed_fallback"

    active_nucleus_mask = markers > 0
    soma_support = (
        active_nucleus_mask
        if soma_expansion == 0
        else ndi.binary_dilation(active_nucleus_mask, iterations=soma_expansion)
    )
    domain = foreground | soma_support
    labels = np.where(domain, assignment, 0).astype(np.int32)
    labels[boundary >= boundary_threshold] = 0

    retained_old_ids: list[int] = []
    for cell_id in sorted(preliminary_mapping):
        cell_mask = labels == cell_id
        gfap_area = int(np.logical_and(cell_mask, foreground).sum())
        if int(cell_mask.sum()) >= min_cell_area and gfap_area >= min_gfap_area:
            retained_old_ids.append(cell_id)
    if not retained_old_ids:
        raise ValueError("No astrocyte instance passed the configured area thresholds")

    relabeled = np.zeros(labels.shape, dtype=np.uint32)
    nucleus_cell_labels = np.zeros(labels.shape, dtype=np.uint32)
    cell_to_nucleus: dict[int, int] = {}
    for new_id, old_id in enumerate(retained_old_ids, start=1):
        nucleus_id = preliminary_mapping[old_id]
        relabeled[labels == old_id] = new_id
        nucleus_cell_labels[nuclei == nucleus_id] = new_id
        relabeled[nuclei == nucleus_id] = new_id
        cell_to_nucleus[new_id] = nucleus_id

    if semantic_probabilities is not None:
        semantic_probs = np.asarray(semantic_probabilities)
        if semantic_probs.shape != (4, *probability.shape):
            raise ValueError("semantic_probabilities must have shape [4, H, W]")
        if not np.isfinite(semantic_probs).all() or np.any(semantic_probs < 0):
            raise ValueError("semantic_probabilities must be finite and non-negative")
        compartments = semantic_probs.argmax(axis=0).astype(np.uint8)
        compartments[(relabeled > 0) & (compartments == 0)] = COMPARTMENT_CLASSES["process"]
        compartments[relabeled == 0] = COMPARTMENT_CLASSES["background"]
    else:
        distance_to_nucleus = ndi.distance_transform_edt(nucleus_cell_labels == 0)
        compartments = np.zeros(probability.shape, dtype=np.uint8)
        compartments[relabeled > 0] = COMPARTMENT_CLASSES["process"]
        compartments[(relabeled > 0) & (distance_to_nucleus <= soma_radius)] = COMPARTMENT_CLASSES["soma"]
    compartments[nucleus_cell_labels > 0] = COMPARTMENT_CLASSES["nucleus"]

    unassigned = np.logical_and(foreground, relabeled == 0)
    return AstrocyteInstanceResult(
        labels=relabeled,
        compartments=compartments,
        nucleus_cell_labels=nucleus_cell_labels,
        cell_to_nucleus=cell_to_nucleus,
        cell_count=len(cell_to_nucleus),
        active_nucleus_count=len(preliminary_mapping),
        unassigned_foreground_fraction=float(unassigned.sum() / foreground.sum()),
        ownership_mode=ownership_mode,
    )
