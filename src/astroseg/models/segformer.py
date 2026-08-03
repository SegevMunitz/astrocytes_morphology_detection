"""Reserved SegFormer integration point."""

from torch import nn


def build_segformer(*_: object, **__: object) -> nn.Module:
    """Raise until a scientifically appropriate SegFormer baseline is implemented."""
    raise NotImplementedError("SegFormer is not implemented; use architecture='unet' for the baseline.")

