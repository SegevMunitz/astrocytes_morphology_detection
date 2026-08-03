"""Lazy patch dataset for astrocyte semantic segmentation."""

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import tifffile
import torch
from torch.utils.data import Dataset

from astroseg.io.manifest import load_manifest, validate_manifest
from astroseg.io.ome_tiff import MicroscopyImage, get_channel, load_ome_tiff
from astroseg.preprocessing.distance_maps import create_nucleus_proximity_map
from astroseg.preprocessing.normalize import percentile_normalize
from astroseg.preprocessing.nuclei import labels_to_binary_mask, validate_nucleus_labels
from astroseg.preprocessing.patches import PatchCoordinates, extract_patch, generate_patch_coordinates

Augmentation = Callable[[np.ndarray, np.ndarray], tuple[np.ndarray, np.ndarray]]


def collate_segmentation_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Collate tensor fields while retaining coordinates as immutable objects."""
    if not batch:
        raise ValueError("Cannot collate an empty batch")
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "target": torch.stack([item["target"] for item in batch]),
        "image_id": [item["image_id"] for item in batch],
        "coordinates": [item["coordinates"] for item in batch],
    }


def _resolve_existing_path(value: str, base_directory: Path, field_name: str) -> Path:
    path = Path(value)
    candidates = [path] if path.is_absolute() else [path, base_directory / path]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{field_name} file does not exist: {value}")


def _load_2d_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        array = np.load(path, allow_pickle=False)
    elif path.suffix.lower() in {".tif", ".tiff"}:
        array = tifffile.imread(path)
    else:
        raise ValueError(f"Unsupported array format {path.suffix!r} for {path}")
    if array.ndim != 2:
        raise ValueError(f"Expected a 2D array in {path}; received shape {array.shape}")
    return array


def prepare_model_inputs(
    microscopy_image: MicroscopyImage,
    gfap_channel: str,
    nucleus_labels: np.ndarray,
    max_nucleus_distance: float = 64.0,
) -> np.ndarray:
    """Build normalized GFAP, nucleus-mask, and nucleus-proximity channels."""
    gfap = get_channel(microscopy_image, gfap_channel)
    validate_nucleus_labels(nucleus_labels, gfap.shape)
    nucleus_mask = labels_to_binary_mask(nucleus_labels)
    proximity = create_nucleus_proximity_map(nucleus_mask, max_nucleus_distance)
    return np.stack((percentile_normalize(gfap), nucleus_mask, proximity)).astype(np.float32)


class AstrocyteDataset(Dataset[dict[str, Any]]):
    """Return aligned three-channel image patches and integer segmentation targets."""

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        patch_size: int = 512,
        overlap: int = 64,
        max_nucleus_distance: float = 64.0,
        augmentation: Augmentation | None = None,
    ) -> None:
        """Validate manifest files and index patches for one data split."""
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train, val, or test")
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
            validate_manifest(frame)
            base_directory = Path.cwd()
        else:
            manifest_path = Path(manifest)
            frame = load_manifest(manifest_path)
            base_directory = manifest_path.parent
        self.manifest = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError(f"Manifest contains no rows for split {split!r}")
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_nucleus_distance = max_nucleus_distance
        self.augmentation = augmentation
        self._records: list[dict[str, Any]] = []
        self._cache_index: int | None = None
        self._cache_value: tuple[np.ndarray, np.ndarray] | None = None

        for row_index, row in self.manifest.iterrows():
            if not row["gfap_channel"].strip():
                raise ValueError(f"gfap_channel is empty for image {row['image_id']!r}")
            for field in ("path", "cellpose_mask_path", "annotation_path"):
                if not row[field].strip():
                    raise ValueError(f"{field} is empty for image {row['image_id']!r}")
            image_path = _resolve_existing_path(row["path"], base_directory, "image")
            nucleus_path = _resolve_existing_path(row["cellpose_mask_path"], base_directory, "Cellpose mask")
            annotation_path = _resolve_existing_path(row["annotation_path"], base_directory, "annotation")
            microscopy = load_ome_tiff(image_path)
            image_shape = microscopy.image.shape[-2:]
            labels = _load_2d_array(nucleus_path)
            target = _load_2d_array(annotation_path)
            validate_nucleus_labels(labels, image_shape)
            if target.shape != image_shape:
                raise ValueError(
                    f"Annotation shape {target.shape} does not match image shape {image_shape} "
                    f"for {row['image_id']!r}"
                )
            if np.any(target < 0) or not np.equal(target, np.floor(target)).all():
                raise ValueError(f"Annotation for {row['image_id']!r} must contain non-negative class integers")
            coordinates = generate_patch_coordinates(image_shape, patch_size, overlap)
            self._records.append(
                {
                    "row_index": row_index,
                    "image_path": image_path,
                    "nucleus_path": nucleus_path,
                    "annotation_path": annotation_path,
                    "coordinates": coordinates,
                }
            )

        self._patch_index = [
            (record_index, coordinates)
            for record_index, record in enumerate(self._records)
            for coordinates in record["coordinates"]
        ]

    def __len__(self) -> int:
        """Return the number of indexed patches."""
        return len(self._patch_index)

    def _load_record(self, record_index: int) -> tuple[np.ndarray, np.ndarray]:
        if self._cache_index == record_index and self._cache_value is not None:
            return self._cache_value
        record = self._records[record_index]
        row = self.manifest.iloc[record["row_index"]]
        microscopy = load_ome_tiff(record["image_path"])
        labels = _load_2d_array(record["nucleus_path"])
        inputs = prepare_model_inputs(
            microscopy,
            row["gfap_channel"],
            labels,
            self.max_nucleus_distance,
        )
        target = _load_2d_array(record["annotation_path"]).astype(np.int64)
        self._cache_index = record_index
        self._cache_value = (inputs, target)
        return inputs, target

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one image patch, target patch, image ID, and coordinates."""
        record_index, coordinates = self._patch_index[index]
        inputs, target = self._load_record(record_index)
        image_patch = extract_patch(inputs, coordinates)
        target_patch = extract_patch(target, coordinates)
        if self.augmentation is not None:
            image_patch, target_patch = self.augmentation(image_patch, target_patch)
        row_index = self._records[record_index]["row_index"]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_patch)).to(dtype=torch.float32),
            "target": torch.from_numpy(np.ascontiguousarray(target_patch)).to(dtype=torch.long),
            "image_id": str(self.manifest.iloc[row_index]["image_id"]),
            "coordinates": coordinates,
        }
