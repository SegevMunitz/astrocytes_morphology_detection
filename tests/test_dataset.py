"""Synthetic dataset and preprocessing tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from astroseg.constants import MANIFEST_COLUMNS
from astroseg.datasets import AstrocyteDataset
from astroseg.preprocessing.distance_maps import create_nucleus_proximity_map
from astroseg.preprocessing.normalize import percentile_normalize
from astroseg.preprocessing.nuclei import labels_to_binary_mask, validate_nucleus_labels


def _write_dataset_files(tmp_path: Path) -> Path:
    image_path = tmp_path / "image.ome.tiff"
    labels_path = tmp_path / "labels.npy"
    annotation_path = tmp_path / "annotation.tiff"
    gfap = np.arange(24 * 20, dtype=np.uint16).reshape(24, 20)
    dapi = np.flipud(gfap)
    tifffile.imwrite(
        image_path,
        np.stack((gfap, dapi)),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["GFAP", "DAPI"]}},
    )
    labels = np.zeros((24, 20), dtype=np.int32)
    labels[9:14, 8:13] = 1
    np.save(labels_path, labels, allow_pickle=False)
    annotation = (gfap > np.median(gfap)).astype(np.uint8)
    tifffile.imwrite(annotation_path, annotation)
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "image_id": "image",
            "path": str(image_path),
            "gfap_channel": "GFAP",
            "dapi_channel": "DAPI",
            "cellpose_mask_path": str(labels_path),
            "annotation_path": str(annotation_path),
            "split": "train",
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    return manifest_path


def test_dataset_returns_expected_tensor_shapes(tmp_path: Path) -> None:
    """One manifest row becomes aligned image and target tensors."""
    dataset = AstrocyteDataset(_write_dataset_files(tmp_path), "train", patch_size=16, overlap=4)
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["target"].shape == (16, 16)
    assert item["image"].dtype == torch.float32
    assert item["target"].dtype == torch.long
    assert item["image_id"] == "image"
    assert torch.all((item["image"] >= 0) & (item["image"] <= 1))


def test_percentile_normalization_range_and_constant_handling() -> None:
    """Normalization is float32, bounded, and safe for constant images."""
    normalized = percentile_normalize(np.arange(100, dtype=np.uint16).reshape(10, 10))
    assert normalized.dtype == np.float32
    assert normalized.min() == 0.0 and normalized.max() == 1.0
    assert not percentile_normalize(np.ones((3, 4), dtype=np.uint16)).any()
    with pytest.raises(ValueError, match="percentiles"):
        percentile_normalize(np.ones((3, 4)), 99, 1)


def test_nucleus_validation_binary_mask_and_proximity() -> None:
    """Nucleus conversions validate alignment and yield bounded float32 inputs."""
    labels = np.zeros((9, 11), dtype=np.int32)
    labels[4, 5] = 3
    validate_nucleus_labels(labels, (9, 11))
    binary = labels_to_binary_mask(labels)
    proximity = create_nucleus_proximity_map(binary, max_distance=4)
    assert binary.dtype == np.float32 and set(np.unique(binary)) == {0.0, 1.0}
    assert proximity.dtype == np.float32
    assert 0.0 <= proximity.min() <= proximity.max() <= 1.0
    assert proximity[4, 5] == 1.0
    with pytest.raises(ValueError, match="does not match"):
        validate_nucleus_labels(labels, (9, 10))
    labels[0, 0] = -1
    with pytest.raises(ValueError, match="negative"):
        validate_nucleus_labels(labels, labels.shape)


def test_proximity_without_nuclei_is_zero() -> None:
    """An empty nucleus mask has zero proximity everywhere."""
    assert not create_nucleus_proximity_map(np.zeros((5, 6), dtype=np.uint8)).any()

