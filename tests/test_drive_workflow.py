"""Tests for the ignored local workspace used by the Google Drive workflow."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage.io import imsave

from scripts.build_drive_manifest import build_drive_manifest


def test_build_drive_manifest_assigns_explicit_train_and_test_splits(
    tmp_path: Path,
) -> None:
    """Drive folder membership should become manifest split metadata.

    Mixed BMP and TIFF inputs are cataloged together while image stems remain
    stable for matching Cellpose ``*_seg.npy`` annotations.
    """
    training = tmp_path / "training"
    test = tmp_path / "test"
    training.mkdir()
    test.mkdir()
    tifffile.imwrite(training / "train_a.tif", np.zeros((4, 5), dtype=np.uint8))
    imsave(
        test / "test_b.bmp",
        np.zeros((4, 5, 3), dtype=np.uint8),
        check_contrast=False,
    )

    output = tmp_path / "metadata" / "manifest.csv"
    manifest = build_drive_manifest(training, test, output)

    assert output.is_file()
    assert manifest.set_index("image_id")["split"].to_dict() == {
        "train_a": "train",
        "test_b": "test",
    }
    assert set(manifest["annotation_status"]) == {"none"}


def test_build_drive_manifest_rejects_duplicate_image_stems(tmp_path: Path) -> None:
    """A training image and test image cannot silently share one stable ID.

    Rejecting the collision prevents a mask from being attached to the wrong
    microscopy image later in the annotation import.
    """
    training = tmp_path / "training"
    test = tmp_path / "test"
    training.mkdir()
    test.mkdir()
    tifffile.imwrite(training / "same.tif", np.zeros((3, 3), dtype=np.uint8))
    imsave(
        test / "same.bmp",
        np.zeros((3, 3, 3), dtype=np.uint8),
        check_contrast=False,
    )

    with pytest.raises(ValueError, match="Duplicate image_id"):
        build_drive_manifest(training, test, tmp_path / "manifest.csv")
