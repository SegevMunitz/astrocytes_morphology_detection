"""Segmentation model definitions."""

from astroseg.models.model_factory import build_model
from astroseg.models.unet import UNet

__all__ = ["UNet", "build_model"]

