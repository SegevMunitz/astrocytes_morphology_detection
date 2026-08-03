"""Patch coverage and exact probability reconstruction tests."""

import numpy as np
import pytest

from astroseg.preprocessing.patches import (
    extract_patch,
    generate_patch_coordinates,
    stitch_probability_patches,
)


def test_patch_coordinates_cover_complete_image() -> None:
    """Every pixel is covered and every coordinate remains inside image bounds."""
    shape = (37, 53)
    coordinates = generate_patch_coordinates(shape, patch_size=16, overlap=5)
    coverage = np.zeros(shape, dtype=np.uint8)
    for coord in coordinates:
        assert coord.y >= 0 and coord.x >= 0
        assert coord.y + coord.height <= shape[0]
        assert coord.x + coord.width <= shape[1]
        coverage[coord.y : coord.y + coord.height, coord.x : coord.x + coord.width] += 1
    assert np.all(coverage > 0)
    assert coordinates == sorted(coordinates, key=lambda item: (item.y, item.x))


def test_stitching_reconstructs_source_probabilities() -> None:
    """Averaging source-valued overlaps reconstructs a synthetic array exactly."""
    rng = np.random.default_rng(7)
    source = rng.random((3, 31, 47), dtype=np.float32)
    coordinates = generate_patch_coordinates(source.shape[-2:], patch_size=15, overlap=6)
    patches = [extract_patch(source, coord) for coord in coordinates]
    reconstructed = stitch_probability_patches(patches, coordinates, source.shape[-2:])
    np.testing.assert_allclose(reconstructed, source, atol=1e-7)


def test_invalid_overlap_is_rejected() -> None:
    """Overlap cannot eliminate or reverse the patch stride."""
    with pytest.raises(ValueError, match="overlap"):
        generate_patch_coordinates((20, 20), patch_size=10, overlap=10)

