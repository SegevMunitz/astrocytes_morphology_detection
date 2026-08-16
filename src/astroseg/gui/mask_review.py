"""Pure data, rendering, and editing operations for interactive mask review."""

from __future__ import annotations

import colorsys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from skimage.segmentation import find_boundaries, relabel_sequential


IMAGE_SUFFIXES = {".bmp", ".tif", ".tiff"}
MASK_SUFFIXES = {".npy", ".tif", ".tiff"}


def _canonical_id(path: Path) -> str:
    """Map image and Cellpose ``*_seg.npy`` names onto one biological ID."""
    stem = path.stem
    return stem.removesuffix("_seg")


def _index_directory(directory: Path, suffixes: set[str], description: str) -> dict[str, Path]:
    """Index unique supported files by canonical image ID."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{description} directory does not exist: {directory}")
    result: dict[str, Path] = {}
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        image_id = _canonical_id(path)
        if image_id in result:
            raise ValueError(
                f"Several {description} files map to {image_id!r}: "
                f"{result[image_id]} and {path}"
            )
        result[image_id] = path
    if not result:
        raise ValueError(f"No supported {description} files found in {directory}")
    return result


@dataclass(frozen=True)
class MaskReviewDataset:
    """Aligned source images, model masks, and optional manual ground truth."""

    images: dict[str, Path]
    model_masks: dict[str, dict[str, Path]]
    ground_truth: dict[str, Path]

    @classmethod
    def discover(
        cls,
        image_directory: Path,
        model_directories: Mapping[str, Path],
        ground_truth_directory: Path | None = None,
    ) -> MaskReviewDataset:
        """Discover exact filename matches without silently guessing substitutes."""
        if not model_directories:
            raise ValueError("At least one named model mask directory is required")
        images = _index_directory(image_directory, IMAGE_SUFFIXES, "image")
        model_masks: dict[str, dict[str, Path]] = {}
        for raw_name, directory in model_directories.items():
            name = str(raw_name).strip()
            if not name:
                raise ValueError("Model names must be non-empty")
            if name in model_masks:
                raise ValueError(f"Model name is duplicated: {name!r}")
            indexed = _index_directory(directory, MASK_SUFFIXES, f"{name} mask")
            model_masks[name] = {
                image_id: path for image_id, path in indexed.items() if image_id in images
            }
            if not model_masks[name]:
                raise ValueError(f"No {name!r} masks match images in {image_directory}")
        ground_truth: dict[str, Path] = {}
        if ground_truth_directory is not None:
            indexed = _index_directory(
                ground_truth_directory, MASK_SUFFIXES, "ground-truth mask"
            )
            ground_truth = {
                image_id: path for image_id, path in indexed.items() if image_id in images
            }
        visible_ids = sorted(
            image_id
            for image_id in images
            if any(image_id in masks for masks in model_masks.values())
        )
        if not visible_ids:
            raise ValueError("No image has a matching model mask")
        return cls(
            images={image_id: images[image_id] for image_id in visible_ids},
            model_masks=model_masks,
            ground_truth=ground_truth,
        )

    @property
    def image_ids(self) -> list[str]:
        """Return deterministic image navigation order."""
        return list(self.images)

    @property
    def model_names(self) -> list[str]:
        """Return model display order from the command line."""
        return list(self.model_masks)


def load_instance_mask(path: str | Path) -> np.ndarray:
    """Load a plain TIFF/NumPy mask or a trusted Cellpose ``*_seg.npy`` export."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Instance mask does not exist: {source}")
    if source.suffix.lower() == ".npy":
        payload = np.load(source, allow_pickle=source.name.endswith("_seg.npy"))
        if payload.dtype == object:
            try:
                item = payload.item()
            except ValueError as exception:
                raise ValueError(f"Invalid Cellpose mask container: {source}") from exception
            if not isinstance(item, dict) or "masks" not in item:
                raise ValueError(f"Cellpose mask has no 'masks' array: {source}")
            array = np.asarray(item["masks"])
        else:
            array = np.asarray(payload)
    else:
        array = np.asarray(tifffile.imread(source))
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"Instance mask must be a non-empty 2D array: {source}")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"Instance mask must contain finite numeric labels: {source}")
    if np.any(array < 0) or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"Instance mask must contain non-negative integers: {source}")
    return array.astype(np.uint32, copy=False)


def _normalized_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert one microscopy plane to robust uint8 grayscale."""
    array = np.asarray(image, dtype=np.float32)
    if array.ndim != 2 or not np.isfinite(array).all():
        raise ValueError("Display image must be a finite 2D array")
    low, high = np.percentile(array, (1.0, 99.0))
    if high <= low:
        return np.zeros(array.shape, dtype=np.uint8)
    normalized = np.clip((array - low) / (high - low), 0.0, 1.0)
    return np.round(normalized * 255).astype(np.uint8)


def _instance_color_table(labels: np.ndarray) -> np.ndarray:
    """Build deterministic Cellpose-style HSV colors for the labels present."""
    maximum = int(labels.max(initial=0))
    table = np.zeros((maximum + 1, 3), dtype=np.uint8)
    for label_id in np.unique(labels[labels > 0]):
        hue = (int(label_id) * 0.61803398875) % 1.0
        rgb = colorsys.hsv_to_rgb(hue, 0.9, 1.0)
        table[int(label_id)] = np.round(np.asarray(rgb) * 255).astype(np.uint8)
    return table


def render_instance_overlay(
    image: np.ndarray,
    labels: np.ndarray | None,
    alpha: float = 0.55,
    show_outlines: bool = True,
    selected_label: int = 0,
) -> np.ndarray:
    """Render a Cellpose-like RGB overlay suitable for Tk or saved comparisons."""
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be in [0, 1]")
    gray = _normalized_grayscale(image)
    base = np.repeat(gray[..., None], 3, axis=2)
    if labels is None:
        return base
    instances = np.asarray(labels)
    if instances.shape != gray.shape:
        raise ValueError("Display image and instance labels must be aligned")
    if np.any(instances < 0) or not np.equal(instances, np.floor(instances)).all():
        raise ValueError("Instance labels must contain non-negative integers")
    instances = instances.astype(np.int64, copy=False)
    colors = _instance_color_table(instances)[instances]
    positive = instances > 0
    overlay = base.astype(np.float32)
    overlay[positive] = (
        (1.0 - alpha) * overlay[positive] + alpha * colors[positive]
    )
    if show_outlines:
        overlay[find_boundaries(instances, mode="outer")] = 255
    if selected_label > 0:
        selected = instances == selected_label
        if selected.any():
            selected_boundary = find_boundaries(selected, mode="outer")
            overlay[selected_boundary] = np.array([255, 230, 0])
    return np.clip(overlay, 0, 255).astype(np.uint8)


def paint_instance_disk(
    labels: np.ndarray,
    y: int,
    x: int,
    radius: int,
    label_id: int,
) -> None:
    """Paint one clipped circular brush stroke into an instance array in place."""
    if labels.ndim != 2:
        raise ValueError("Editable labels must be 2D")
    if radius < 1 or label_id < 0:
        raise ValueError("Brush radius must be positive and label_id non-negative")
    height, width = labels.shape
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    disk = (yy - y) ** 2 + (xx - x) ** 2 <= radius**2
    patch = labels[y0:y1, x0:x1]
    patch[disk] = label_id


@dataclass(frozen=True)
class SavedCorrection:
    """Plain training TIFF and optional Cellpose-compatible editable export."""

    tiff_path: Path
    cellpose_path: Path | None
    cell_count: int


def save_corrected_instances(
    image_id: str,
    labels: np.ndarray,
    output_directory: Path,
    source_image_path: Path,
    export_cellpose: bool = True,
    overwrite: bool = False,
    cellpose_channels: tuple[int, int] = (1, 2),
) -> SavedCorrection:
    """Save reviewed labels separately from predictions, never over the source mask."""
    if not image_id.strip():
        raise ValueError("image_id must be non-empty")
    if len(cellpose_channels) != 2 or any(value not in range(4) for value in cellpose_channels):
        raise ValueError("cellpose_channels must contain two values in [0, 3]")
    instances = load_instance_mask_array(labels)
    sequential, _, _ = relabel_sequential(instances)
    sequential = sequential.astype(np.uint32, copy=False)
    output_directory.mkdir(parents=True, exist_ok=True)
    tiff_path = output_directory / f"{image_id}.tiff"
    cellpose_path = output_directory / f"{image_id}_seg.npy" if export_cellpose else None
    destinations = [tiff_path, *([cellpose_path] if cellpose_path is not None else [])]
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite corrected masks: {existing}")
    tifffile.imwrite(tiff_path, sequential, photometric="minisblack")
    ids = np.unique(sequential[sequential > 0])
    if cellpose_path is not None:
        payload = {
            "masks": sequential,
            "outlines": find_boundaries(sequential, mode="inner"),
            "ismanual": np.ones(len(ids), dtype=bool),
            "filename": str(source_image_path.resolve()),
            "chan_choose": list(cellpose_channels),
        }
        np.save(cellpose_path, payload, allow_pickle=True)
    return SavedCorrection(tiff_path, cellpose_path, int(len(ids)))


def load_instance_mask_array(labels: np.ndarray) -> np.ndarray:
    """Validate an in-memory editable instance array before saving."""
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ValueError("Corrected instance labels must be a non-empty 2D array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError("Corrected instance labels must be finite numeric values")
    if np.any(array < 0) or not np.equal(array, np.floor(array)).all():
        raise ValueError("Corrected instance labels must be non-negative integers")
    return array.astype(np.uint32, copy=False)
