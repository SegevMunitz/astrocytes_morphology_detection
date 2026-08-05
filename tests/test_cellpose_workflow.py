import os
from pathlib import Path

import numpy as np
import pytest
import tifffile

from scripts.prepare_cellpose_training_data import stage_pairs


def _write_pair(directory: Path, channels: tuple[int, int] = (1, 3)) -> None:
    """Write one small GUI-style image and instance annotation pair."""
    image = np.zeros((16, 20, 3), dtype=np.uint8)
    masks = np.zeros((16, 20), dtype=np.uint16)
    masks[2:8, 3:9] = 1
    tifffile.imwrite(directory / "example.tif", image)
    np.save(
        directory / "example_seg.npy",
        {"masks": masks, "chan_choose": list(channels), "ismanual": [True]},
        allow_pickle=True,
    )


@pytest.mark.skipif(os.name == "nt", reason="Windows requires elevated symlink privileges")
def test_stage_pairs_validates_and_preserves_sources(tmp_path: Path) -> None:
    """Staging should create symlinks without rewriting the original annotations."""
    source = tmp_path / "source"
    source.mkdir()
    _write_pair(source)
    staging = tmp_path / "run"

    pairs, instances = stage_pairs(source, staging, (1, 3))

    assert (pairs, instances) == (1, 1)
    assert (staging / "example.tif").is_symlink()
    assert (staging / "example_seg.npy").is_symlink()
    assert np.load(source / "example_seg.npy", allow_pickle=True).item()["masks"].max() == 1


def test_stage_pairs_rejects_inconsistent_channels(tmp_path: Path) -> None:
    """A run must not silently mix annotations made with different channels."""
    source = tmp_path / "source"
    source.mkdir()
    _write_pair(source, channels=(2, 3))

    with pytest.raises(ValueError, match="uses channels"):
        stage_pairs(source, tmp_path / "run", (1, 3))


@pytest.mark.skipif(os.name == "nt", reason="Windows requires elevated symlink privileges")
def test_stage_pairs_selects_requested_image_ids(tmp_path: Path) -> None:
    """Explicit split lists should stage only their requested biological images."""
    source = tmp_path / "source"
    source.mkdir()
    _write_pair(source)
    other_image = np.zeros((16, 20, 3), dtype=np.uint8)
    other_mask = np.zeros((16, 20), dtype=np.uint16)
    other_mask[1:4, 1:4] = 1
    tifffile.imwrite(source / "other.tif", other_image)
    np.save(
        source / "other_seg.npy",
        {"masks": other_mask, "chan_choose": [1, 3]},
        allow_pickle=True,
    )

    pairs, instances = stage_pairs(source, tmp_path / "run", (1, 3), {"other"})

    assert (pairs, instances) == (1, 1)
    assert not (tmp_path / "run" / "example.tif").exists()
    assert (tmp_path / "run" / "other.tif").is_symlink()
