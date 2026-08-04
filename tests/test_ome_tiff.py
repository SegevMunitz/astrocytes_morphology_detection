"""Synthetic tests for OME-TIFF loading and channel selection."""

from pathlib import Path

import numpy as np
import pytest
import tifffile
from skimage.io import imsave

from astroseg.io.ome_tiff import MicroscopyImage, get_channel, load_ome_tiff
from scripts.build_manifest import build_manifest


def test_load_ome_tiff_channel_first_and_metadata(tmp_path: Path) -> None:
    """Loader preserves dtype and extracts named channels from synthetic OME XML."""
    path = tmp_path / "synthetic.ome.tiff"
    array = np.arange(2 * 12 * 10, dtype=np.uint16).reshape(2, 12, 10)
    tifffile.imwrite(
        path,
        array,
        ome=True,
        metadata={
            "axes": "CYX",
            "Channel": {"Name": ["GFAP", "DAPI"]},
            "PhysicalSizeX": 0.5,
            "PhysicalSizeXUnit": "µm",
        },
    )
    loaded = load_ome_tiff(path)
    assert loaded.image.shape == (2, 12, 10)
    assert loaded.image.dtype == np.uint16
    assert loaded.channel_names == ["GFAP", "DAPI"]
    assert loaded.pixel_size_um == pytest.approx(0.5)
    np.testing.assert_array_equal(loaded.image, array)


def test_get_channel_exact_then_case_insensitive(tmp_path: Path) -> None:
    """Channel lookup uses exact names without fuzzy matching."""
    image = MicroscopyImage(
        image=np.stack((np.ones((4, 5)), np.full((4, 5), 2))),
        channel_names=["GFAP", "DAPI"],
        pixel_size_um=None,
        source_path=tmp_path / "example.tif",
    )
    assert np.all(get_channel(image, "GFAP") == 1)
    assert np.all(get_channel(image, "dapi") == 2)
    with pytest.raises(KeyError, match="not found"):
        get_channel(image, "GFA")


def test_load_non_ome_rgb_assigns_sample_names(tmp_path: Path) -> None:
    """Standard RGB TIFF samples receive explicit color-channel names."""
    path = tmp_path / "composite.tif"
    array = np.zeros((12, 10, 3), dtype=np.uint8)
    array[..., 1] = 17
    array[..., 2] = 29
    tifffile.imwrite(path, array, photometric="rgb")

    loaded = load_ome_tiff(path)

    assert loaded.image.shape == (3, 12, 10)
    assert loaded.channel_names == ["Red", "Green", "Blue"]
    np.testing.assert_array_equal(get_channel(loaded, "Green"), array[..., 1])
    np.testing.assert_array_equal(get_channel(loaded, "blue"), array[..., 2])


def test_load_rgb_bmp_assigns_color_channels(tmp_path: Path) -> None:
    """RGB BMP input should use the same channel contract as an RGB TIFF.

    BMP metadata has no biological names, so explicit color names remain the
    stable basis for automatic GFAP/DAPI selection.
    """
    path = tmp_path / "ctrl_437.bmp"
    array = np.zeros((13, 11, 3), dtype=np.uint8)
    array[..., 0] = 9
    array[..., 1] = 41
    array[..., 2] = 83
    imsave(path, array, check_contrast=False)

    loaded = load_ome_tiff(path)

    assert loaded.image.shape == (3, 13, 11)
    assert loaded.channel_names == ["Red", "Green", "Blue"]
    assert loaded.pixel_size_um is None
    np.testing.assert_array_equal(get_channel(loaded, "Red"), array[..., 0])
    np.testing.assert_array_equal(get_channel(loaded, "Blue"), array[..., 2])


def test_manifest_discovers_bmp_tif_and_tiff_images(tmp_path: Path) -> None:
    """Dataset discovery should treat every supported microscopy suffix equally.

    Train/test assignment remains manifest metadata and is independent of format.
    """
    raw = tmp_path / "raw"
    raw.mkdir()
    imsave(raw / "ctrl_437.bmp", np.zeros((5, 6, 3), dtype=np.uint8), check_contrast=False)
    tifffile.imwrite(raw / "72h_407.tif", np.zeros((5, 6), dtype=np.uint8))
    tifffile.imwrite(raw / "other.tiff", np.zeros((5, 6), dtype=np.uint8))

    manifest = build_manifest(raw, tmp_path / "manifest.csv")

    assert set(manifest["image_id"]) == {"ctrl_437", "72h_407", "other"}
    assert {Path(value).suffix.lower() for value in manifest["path"]} == {
        ".bmp",
        ".tif",
        ".tiff",
    }


def test_load_rejects_non_singleton_depth(tmp_path: Path) -> None:
    """Loader refuses to silently choose among multiple Z planes."""
    path = tmp_path / "stack.ome.tiff"
    tifffile.imwrite(path, np.zeros((2, 2, 8, 8), dtype=np.uint8), ome=True, metadata={"axes": "ZCYX"})
    with pytest.raises(ValueError, match="non-singleton"):
        load_ome_tiff(path)
