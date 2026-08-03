"""Segmentation model factory."""

from torch import nn

from astroseg.models.segformer import build_segformer
from astroseg.models.unet import UNet


def build_model(
    architecture: str,
    input_channels: int,
    num_classes: int,
    base_channels: int = 32,
) -> nn.Module:
    """Build a supported per-pixel segmentation model."""
    name = architecture.strip().lower()
    if name == "unet":
        return UNet(input_channels, num_classes, base_channels)
    if name == "segformer":
        return build_segformer(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"Unsupported architecture {architecture!r}; supported architecture: unet")

