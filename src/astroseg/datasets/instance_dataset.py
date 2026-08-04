"""Patch dataset for nucleus-guided astrocyte instance segmentation."""

from collections.abc import Callable, Collection
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from astroseg.constants import ANNOTATION_STATUSES, TRAINABLE_ANNOTATION_STATUSES
from astroseg.datasets.astrocyte_dataset import (
    _load_2d_array,
    _resolve_existing_path,
    prepare_model_inputs,
)
from astroseg.io.manifest import load_manifest, validate_manifest
from astroseg.io.ome_tiff import load_ome_tiff
from astroseg.preprocessing.instance_targets import (
    AstrocyteInstanceTargets,
    build_astrocyte_instance_targets,
)
from astroseg.preprocessing.nuclei import validate_nucleus_labels
from astroseg.preprocessing.patches import PatchCoordinates, extract_patch, generate_patch_coordinates

InstanceAugmentation = Callable[
    [np.ndarray, dict[str, np.ndarray]],
    tuple[np.ndarray, dict[str, np.ndarray]],
]


def collate_instance_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Stack multi-head tensors while preserving reconstruction metadata.

    Targets remain under a dedicated mapping consumed by the instance loss, while
    full instance IDs are retained for later object-level evaluation.
    """
    if not batch:
        raise ValueError("Cannot collate an empty instance batch")
    return {
        "image": torch.stack([item["image"] for item in batch]),
        "targets": {
            name: torch.stack([item["targets"][name] for item in batch])
            for name in ("semantic", "boundary", "offsets", "offset_mask")
        },
        "instances": torch.stack([item["instances"] for item in batch]),
        "image_id": [item["image_id"] for item in batch],
        "coordinates": [item["coordinates"] for item in batch],
    }


class AstrocyteInstanceDataset(Dataset[dict[str, Any]]):
    """Provide inputs and ownership-aware targets for individual astrocytes.

    Rows require a full-cell instance label image in addition to nucleus labels.
    Optional explicit compartment annotations override derived soma/process targets.
    """

    def __init__(
        self,
        manifest: str | Path | pd.DataFrame,
        split: str,
        patch_size: int = 512,
        overlap: int = 64,
        max_nucleus_distance: float = 64.0,
        soma_radius: float = 20.0,
        offset_scale: float = 256.0,
        augmentation: InstanceAugmentation | None = None,
        annotation_statuses: Collection[str] | None = TRAINABLE_ANNOTATION_STATUSES,
        manifest_base_directory: str | Path | None = None,
    ) -> None:
        """Validate aligned instance files and index deterministic image patches.

        Automatic pseudo rows remain excluded by default. Human lifecycle status
        alone is insufficient: each selected row must also provide instance labels.
        """
        if split not in {"train", "val", "test"}:
            raise ValueError("split must be one of train, val, or test")
        if isinstance(manifest, pd.DataFrame):
            frame = manifest.copy()
            validate_manifest(frame)
            base_directory = Path(manifest_base_directory) if manifest_base_directory else Path.cwd()
        else:
            manifest_path = Path(manifest)
            frame = load_manifest(manifest_path)
            base_directory = manifest_path.parent
        selected = frame["split"] == split
        if annotation_statuses is not None:
            if isinstance(annotation_statuses, str):
                raise TypeError("annotation_statuses must be a collection, not one string")
            normalized = {str(status).strip().lower() for status in annotation_statuses}
            invalid = normalized - ANNOTATION_STATUSES
            if invalid:
                raise ValueError(f"Invalid dataset annotation statuses: {sorted(invalid)}")
            selected &= frame["annotation_status"].isin(normalized)
        self.manifest = frame.loc[selected].reset_index(drop=True)
        if self.manifest.empty:
            raise ValueError(f"Manifest contains no eligible instance rows for split {split!r}")
        self.patch_size = patch_size
        self.overlap = overlap
        self.max_nucleus_distance = max_nucleus_distance
        self.soma_radius = soma_radius
        self.offset_scale = offset_scale
        self.augmentation = augmentation
        self._records: list[dict[str, Any]] = []
        self._cache_index: int | None = None
        self._cache_value: tuple[np.ndarray, AstrocyteInstanceTargets] | None = None

        for row_index, row in self.manifest.iterrows():
            for field in ("path", "gfap_channel", "cellpose_mask_path", "instance_annotation_path"):
                if not str(row[field]).strip():
                    raise ValueError(f"{field} is empty for image {row['image_id']!r}")
            image_path = _resolve_existing_path(str(row["path"]), base_directory, "image")
            nucleus_path = _resolve_existing_path(
                str(row["cellpose_mask_path"]), base_directory, "nucleus mask"
            )
            instance_path = _resolve_existing_path(
                str(row["instance_annotation_path"]), base_directory, "instance annotation"
            )
            compartment_path = None
            if str(row["compartment_annotation_path"]).strip():
                compartment_path = _resolve_existing_path(
                    str(row["compartment_annotation_path"]),
                    base_directory,
                    "compartment annotation",
                )
            microscopy = load_ome_tiff(image_path)
            nuclei = _load_2d_array(nucleus_path)
            cells = _load_2d_array(instance_path)
            compartments = _load_2d_array(compartment_path) if compartment_path else None
            validate_nucleus_labels(nuclei, microscopy.image.shape[-2:])
            build_astrocyte_instance_targets(
                cells,
                nuclei,
                compartments,
                soma_radius=soma_radius,
                offset_scale=offset_scale,
                max_nucleus_distance=max_nucleus_distance,
            )
            coordinates = generate_patch_coordinates(
                microscopy.image.shape[-2:], patch_size, overlap
            )
            self._records.append(
                {
                    "row_index": row_index,
                    "image_path": image_path,
                    "nucleus_path": nucleus_path,
                    "instance_path": instance_path,
                    "compartment_path": compartment_path,
                    "coordinates": coordinates,
                }
            )
        self._patch_index = [
            (record_index, coordinate)
            for record_index, record in enumerate(self._records)
            for coordinate in record["coordinates"]
        ]

    def __len__(self) -> int:
        """Return the deterministic number of patches across selected images.

        No pixels are loaded by this operation; complete image records remain lazy.
        """
        return len(self._patch_index)

    def _load_record(self, record_index: int) -> tuple[np.ndarray, AstrocyteInstanceTargets]:
        """Load and cache one complete input/target record for adjacent patches.

        Target construction occurs before cropping so nucleus-directed offsets use
        full-image coordinates consistently across overlapping patches.
        """
        if self._cache_index == record_index and self._cache_value is not None:
            return self._cache_value
        record = self._records[record_index]
        row = self.manifest.iloc[record["row_index"]]
        microscopy = load_ome_tiff(record["image_path"])
        nuclei = _load_2d_array(record["nucleus_path"])
        cells = _load_2d_array(record["instance_path"])
        compartments = (
            _load_2d_array(record["compartment_path"])
            if record["compartment_path"] is not None
            else None
        )
        inputs = prepare_model_inputs(
            microscopy, str(row["gfap_channel"]), nuclei, self.max_nucleus_distance
        )
        targets = build_astrocyte_instance_targets(
            cells,
            nuclei,
            compartments,
            soma_radius=self.soma_radius,
            offset_scale=self.offset_scale,
            max_nucleus_distance=self.max_nucleus_distance,
        )
        self._cache_index = record_index
        self._cache_value = (inputs, targets)
        return inputs, targets

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one image patch and its complete multi-head supervision mapping.

        Semantic/boundary labels are integer tensors, offsets are float vectors,
        and original instance IDs support object-level evaluation after stitching.
        """
        record_index, coordinate = self._patch_index[index]
        inputs, targets = self._load_record(record_index)
        image_patch = extract_patch(inputs, coordinate)
        target_arrays = {
            "semantic": extract_patch(targets.semantic, coordinate),
            "boundary": extract_patch(targets.boundary, coordinate),
            "offsets": extract_patch(targets.offsets, coordinate),
            "offset_mask": extract_patch(targets.offset_mask, coordinate),
            "instances": extract_patch(targets.instances, coordinate),
        }
        if self.augmentation is not None:
            image_patch, target_arrays = self.augmentation(image_patch, target_arrays)
        row_index = self._records[record_index]["row_index"]
        return {
            "image": torch.from_numpy(np.ascontiguousarray(image_patch)).float(),
            "targets": {
                "semantic": torch.from_numpy(target_arrays["semantic"]).long(),
                "boundary": torch.from_numpy(target_arrays["boundary"]).long(),
                "offsets": torch.from_numpy(target_arrays["offsets"]).float(),
                "offset_mask": torch.from_numpy(target_arrays["offset_mask"]).float(),
            },
            "instances": torch.from_numpy(target_arrays["instances"].astype(np.int64)),
            "image_id": str(self.manifest.iloc[row_index]["image_id"]),
            "coordinates": coordinate,
        }
