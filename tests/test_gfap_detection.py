"""Tests for heuristic GFAP bootstrap pseudo-label generation."""

import numpy as np

from astroseg.preprocessing import detect_gfap_bootstrap_mask


def test_gfap_bootstrap_retains_connected_dim_process() -> None:
    """A dim process connected to a bright soma survives hysteresis filtering.

    An equally dim isolated patch is rejected because it has no connection to
    strong GFAP signal, demonstrating the purpose of the two thresholds.
    """
    gfap = np.zeros((96, 96), dtype=np.uint16)
    gfap[40:56, 40:56] = 4000
    gfap[46:50, 18:40] = 1200
    gfap[70:76, 70:80] = 1200

    result = detect_gfap_bootstrap_mask(
        gfap,
        gaussian_sigma=0.5,
        low_threshold_ratio=0.4,
        min_component_area=12,
        max_hole_area=0,
    )

    assert result.mask[48, 25] == 1
    assert result.mask[48, 48] == 1
    assert result.mask[73, 75] == 0
    assert result.probabilities.shape == (2, 96, 96)
    np.testing.assert_allclose(result.probabilities.sum(axis=0), 1.0)
    np.testing.assert_array_equal(result.probabilities.argmax(axis=0), result.mask)


def test_gfap_bootstrap_rejects_constant_image() -> None:
    """A channel without intensity variation cannot produce a pseudo target."""
    with np.testing.assert_raises_regex(ValueError, "no detectable intensity"):
        detect_gfap_bootstrap_mask(np.ones((32, 32), dtype=np.uint8))
