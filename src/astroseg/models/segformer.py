"""Reserved SegFormer integration point."""

from torch import nn


def build_segformer(*_: object, **__: object) -> nn.Module:
    """Mark the reserved SegFormer integration point as unavailable.

    Keeping the failure explicit prevents configuration files from silently
    selecting an unimplemented architecture or an unsuitable external default.
    """
    raise NotImplementedError("SegFormer is not implemented; use architecture='unet' for the baseline.")
