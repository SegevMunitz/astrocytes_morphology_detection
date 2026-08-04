"""Overlap-averaged inference for all nucleus-guided instance heads."""

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from astroseg.preprocessing.patches import (
    extract_patch,
    generate_patch_coordinates,
    stitch_probability_patches,
)


@dataclass(frozen=True)
class InstanceHeadPredictions:
    """Full-resolution compartment, boundary, and ownership predictions.

    Compartment probabilities have four channels, boundary probability is the
    positive class plane, and offsets store normalized ``dy, dx`` vectors.
    """

    semantic_probabilities: np.ndarray
    boundary_probability: np.ndarray
    ownership_offsets: np.ndarray


def predict_instance_patch(
    model: nn.Module, inputs: np.ndarray, device: torch.device
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Run all model heads on one channel-first patch without gradients.

    The returned arrays omit the temporary batch dimension and are float32 NumPy
    values ready for overlap averaging.
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [C, H, W]; received {inputs.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(inputs))[None].to(
        device=device, dtype=torch.float32
    )
    model.eval()
    with torch.inference_mode():
        outputs = model(tensor)
        required = {"semantic_logits", "boundary_logits", "offsets"}
        if not isinstance(outputs, dict) or required - set(outputs):
            raise ValueError("Model does not expose the required instance prediction heads")
        semantic = torch.softmax(outputs["semantic_logits"], dim=1)[0]
        boundary = torch.softmax(outputs["boundary_logits"], dim=1)[0, 1:2]
        offsets = outputs["offsets"][0]
    return tuple(
        value.cpu().numpy().astype(np.float32, copy=False)
        for value in (semantic, boundary, offsets)
    )


def predict_instance_full_image(
    model: nn.Module,
    inputs: np.ndarray,
    patch_size: int,
    overlap: int,
    device: torch.device,
) -> InstanceHeadPredictions:
    """Predict and reconstruct all heads over one complete microscopy image.

    Overlap regions are averaged for probabilities and continuous offset vectors,
    avoiding seams from independently thresholded patches.
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [C, H, W]; received {inputs.shape}")
    coordinates = generate_patch_coordinates(inputs.shape[-2:], patch_size, overlap)
    predictions = [
        predict_instance_patch(model, extract_patch(inputs, coordinate), device)
        for coordinate in coordinates
    ]
    semantic = stitch_probability_patches(
        [value[0] for value in predictions], coordinates, inputs.shape[-2:]
    )
    boundary = stitch_probability_patches(
        [value[1] for value in predictions], coordinates, inputs.shape[-2:]
    )[0]
    offsets = stitch_probability_patches(
        [value[2] for value in predictions], coordinates, inputs.shape[-2:]
    )
    return InstanceHeadPredictions(semantic, boundary, offsets)
