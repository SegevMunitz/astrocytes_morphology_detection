"""Single-patch model inference."""

import numpy as np
import torch
from torch import nn


def predict_patch(model: nn.Module, inputs: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference on one channel-first patch and return class probabilities.

    A batch dimension is added temporarily, gradients are disabled, and softmax
    converts per-pixel logits to a float32 ``[num_classes, H, W]`` array.
    """
    if inputs.ndim != 3:
        raise ValueError(f"inputs must have shape [C, H, W]; received {inputs.shape}")
    tensor = torch.from_numpy(np.ascontiguousarray(inputs)).unsqueeze(0).to(device=device, dtype=torch.float32)
    model.eval()
    with torch.inference_mode():
        logits = model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    return probabilities.cpu().numpy().astype(np.float32, copy=False)
