"""Deterministic patch extraction and overlap-aware stitching."""

from dataclasses import dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class PatchCoordinates:
    """Immutable rectangular patch coordinates."""

    y: int
    x: int
    height: int
    width: int


def _starts(length: int, patch_size: int, stride: int) -> list[int]:
    if length <= patch_size:
        return [0]
    starts = list(range(0, length - patch_size + 1, stride))
    final = length - patch_size
    if starts[-1] != final:
        starts.append(final)
    return starts


def generate_patch_coordinates(
    image_shape: tuple[int, int],
    patch_size: int = 512,
    overlap: int = 64,
) -> list[PatchCoordinates]:
    """Generate row-major coordinates that completely cover an image."""
    if len(image_shape) != 2 or any(not isinstance(v, int) or v <= 0 for v in image_shape):
        raise ValueError(f"image_shape must contain two positive integers; got {image_shape}")
    if not isinstance(patch_size, int) or patch_size <= 0:
        raise ValueError("patch_size must be a positive integer")
    if not isinstance(overlap, int) or overlap < 0 or overlap >= patch_size:
        raise ValueError("overlap must satisfy 0 <= overlap < patch_size")
    height, width = image_shape
    patch_height = min(patch_size, height)
    patch_width = min(patch_size, width)
    stride = patch_size - overlap
    return [
        PatchCoordinates(y=y, x=x, height=patch_height, width=patch_width)
        for y in _starts(height, patch_height, min(stride, patch_height))
        for x in _starts(width, patch_width, min(stride, patch_width))
    ]


def extract_patch(array: np.ndarray, coordinates: PatchCoordinates) -> np.ndarray:
    """Extract a patch from the final two dimensions of an array."""
    if array.ndim < 2:
        raise ValueError("array must have at least two dimensions")
    if min(coordinates.y, coordinates.x, coordinates.height, coordinates.width) < 0:
        raise ValueError("patch coordinates must be non-negative")
    if coordinates.height == 0 or coordinates.width == 0:
        raise ValueError("patch height and width must be positive")
    y2 = coordinates.y + coordinates.height
    x2 = coordinates.x + coordinates.width
    if y2 > array.shape[-2] or x2 > array.shape[-1]:
        raise ValueError(f"Patch {coordinates} lies outside array shape {array.shape[-2:]}")
    return array[..., coordinates.y:y2, coordinates.x:x2]


def stitch_probability_patches(
    patches: Sequence[np.ndarray],
    coordinates: Sequence[PatchCoordinates],
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Average overlapping ``[C, H, W]`` probability patches into one image."""
    if len(patches) == 0 or len(patches) != len(coordinates):
        raise ValueError("patches and coordinates must have the same non-zero length")
    first = np.asarray(patches[0])
    if first.ndim != 3:
        raise ValueError("probability patches must have shape [C, H, W]")
    output = np.zeros((first.shape[0], *image_shape), dtype=np.float64)
    counts = np.zeros(image_shape, dtype=np.float64)
    for patch, coord in zip(patches, coordinates, strict=True):
        patch_array = np.asarray(patch)
        expected = (first.shape[0], coord.height, coord.width)
        if patch_array.shape != expected:
            raise ValueError(f"Patch shape {patch_array.shape} does not match coordinates {expected}")
        y2, x2 = coord.y + coord.height, coord.x + coord.width
        output[:, coord.y:y2, coord.x:x2] += patch_array
        counts[coord.y:y2, coord.x:x2] += 1.0
    if np.any(counts == 0):
        raise ValueError("Patch coordinates do not cover the complete image")
    return (output / counts[np.newaxis, ...]).astype(np.float32)
