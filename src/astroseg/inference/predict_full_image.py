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
    """Predict a full channel-first image using overlapping model patches.

    Softmax probabilities—not hard labels—are averaged in overlap regions.
    The returned ``[num_classes, H, W]`` array preserves the source dimensions.
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [C, H, W]; received {inputs.shape}")
    coordinates = generate_patch_coordinates(inputs.shape[-2:], patch_size, overlap)
    patches = [predict_patch(model, extract_patch(inputs, coord), device) for coord in coordinates]
    return stitch_probability_patches(patches, coordinates, inputs.shape[-2:])
