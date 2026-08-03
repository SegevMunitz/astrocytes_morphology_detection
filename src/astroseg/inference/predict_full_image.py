"""Overlap-averaged full-resolution inference."""

import numpy as np
import torch
from torch import nn

from astroseg.inference.predict_patch import predict_patch
from astroseg.preprocessing.patches import (
    extract_patch,
    generate_patch_coordinates,
    stitch_probability_patches,
)


def predict_full_image(
    model: nn.Module,
    inputs: np.ndarray,
    patch_size: int,
    overlap: int,
    device: torch.device,
) -> np.ndarray:
    """Predict and overlap-average a full ``[C, H, W]`` image."""
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [C, H, W]; received {inputs.shape}")
    coordinates = generate_patch_coordinates(inputs.shape[-2:], patch_size, overlap)
    patches = [predict_patch(model, extract_patch(inputs, coord), device) for coord in coordinates]
    return stitch_probability_patches(patches, coordinates, inputs.shape[-2:])

