"""Weight conversion utilities for three-fluorescence-channel Cellpose models."""

from collections.abc import Mapping

import numpy as np
import torch


def prepare_three_channel_cellpose_image(image: np.ndarray) -> np.ndarray:
    """Return contiguous GFAP/GFP/DAPI planes and discard transmitted light.

    The acquisition TIFFs are either ``Cy5, DAPI`` (test), three fluorescence
    planes, or four planes ending in transmitted light, channel-first or last.
    """
    array = np.asarray(image)
    if array.ndim != 3:
        raise ValueError(f"Three-channel Cellpose image must be 3D, received {array.shape}")
    if array.shape[0] in (2, 3, 4) and array.shape[-1] not in (2, 3, 4):
        channel_first = array
    elif array.shape[-1] in (2, 3, 4):
        channel_first = np.moveaxis(array, -1, 0)
    else:
        raise ValueError(f"Expected two to four acquisition channels, received {array.shape}")
    if channel_first.shape[0] == 2:
        result = np.stack(
            (channel_first[0], np.zeros_like(channel_first[0]), channel_first[1])
        )
    else:
        result = channel_first[:3]
    return np.ascontiguousarray(result)


def expand_cellpose_input_weights(
    source_state: Mapping[str, torch.Tensor],
    target_state: Mapping[str, torch.Tensor],
) -> tuple[dict[str, torch.Tensor], tuple[str, ...]]:
    """Map a two-channel Cellpose state onto GFAP/GFP/DAPI input channels.

    Every shape-compatible tensor is copied exactly. The one input convolution is
    expanded so source GFAP goes to channel 0, source DAPI goes to channel 2, and
    the new GFP channel starts at zero. Consequently, an input with zero GFP has
    exactly the same first-layer activation as the source two-channel model.
    """
    if set(source_state) != set(target_state):
        missing = sorted(set(target_state) - set(source_state))
        unexpected = sorted(set(source_state) - set(target_state))
        raise ValueError(
            f"Cellpose state keys differ; missing={missing}, unexpected={unexpected}"
        )
    converted: dict[str, torch.Tensor] = {}
    expanded_keys: list[str] = []
    for name, target in target_state.items():
        source = source_state[name]
        if source.shape == target.shape:
            converted[name] = source.detach().clone()
            continue
        if source.ndim == 1 and source.shape == (2,) and target.shape == (3,):
            expanded = target.detach().clone()
            expanded[0] = source[0].to(device=target.device, dtype=target.dtype)
            expanded[2] = source[1].to(device=target.device, dtype=target.dtype)
            converted[name] = expanded
            continue
        expandable = (
            source.ndim in (4, 5)
            and target.ndim == source.ndim
            and source.shape[0] == target.shape[0]
            and source.shape[1] == 2
            and target.shape[1] == 3
            and source.shape[2:] == target.shape[2:]
        )
        if not expandable:
            raise ValueError(
                f"Unsupported Cellpose tensor shape change for {name!r}: "
                f"{tuple(source.shape)} -> {tuple(target.shape)}"
            )
        expanded = torch.zeros_like(target)
        expanded[:, 0] = source[:, 0].to(device=target.device, dtype=target.dtype)
        expanded[:, 2] = source[:, 1].to(device=target.device, dtype=target.dtype)
        converted[name] = expanded
        expanded_keys.append(name)
    if not expanded_keys:
        raise ValueError(
            "Expected at least one two-to-three-channel Cellpose input tensor"
        )
    return converted, tuple(expanded_keys)
