"""Tests for automatic channel selection and nucleus instance detection."""

from pathlib import Path

import numpy as np
from skimage.draw import disk

from astroseg.io.ome_tiff import MicroscopyImage
from astroseg.preprocessing import detect_nucleus_instances, select_model_channels


def _rgb_image(red_value: int, green_value: int, tmp_path: Path) -> MicroscopyImage:
    """Build a small named RGB microscopy record for channel-selection tests."""
    image = np.zeros((3, 32, 32), dtype=np.uint8)
    image[0, 8:24, 8:24] = red_value
    image[1, 8:24, 8:24] = green_value
    image[2, 10:14, 10:14] = 200
    return MicroscopyImage(image, ["Red", "Green", "Blue"], None, tmp_path / "rgb.tif")


def test_rgb_channel_selection_uses_blue_and_dominant_structure(tmp_path: Path) -> None:
    """RGB inference selects Blue for DAPI and the stronger Red/Green signal."""
    green = select_model_channels(_rgb_image(0, 180, tmp_path))
    red = select_model_channels(_rgb_image(190, 20, tmp_path))

    assert (green.gfap_channel, green.dapi_channel) == ("Green", "Blue")
    assert (red.gfap_channel, red.dapi_channel) == ("Red", "Blue")
    assert green.method == red.method == "rgb_signal"


def test_manifest_channel_selection_takes_priority(tmp_path: Path) -> None:
    """Explicit valid manifest values are retained instead of inferred."""
    selection = select_model_channels(
        _rgb_image(200, 10, tmp_path), gfap_channel="Green", dapi_channel="Blue"
    )
    assert selection.gfap_channel == "Green"
    assert selection.method == "manifest"


def test_detect_nucleus_instances_separates_bright_objects() -> None:
    """Separated synthetic DAPI disks become sequential instance labels."""
    dapi = np.zeros((96, 96), dtype=np.uint16)
    for center in ((28, 30), (65, 62), (28, 70)):
        rr, cc = disk(center, 8, shape=dapi.shape)
        dapi[rr, cc] = 4000

    result = detect_nucleus_instances(dapi, min_nucleus_area=30, min_peak_distance=7)

    assert result.labels.dtype == np.uint32
    assert result.labels.shape == dapi.shape
    assert result.instance_count == 3
    assert set(np.unique(result.labels)) == {0, 1, 2, 3}
    np.testing.assert_array_equal(result.binary_mask, result.labels > 0)
