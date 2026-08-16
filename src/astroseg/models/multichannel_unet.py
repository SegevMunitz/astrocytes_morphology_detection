"""Residual GroupNorm U-Net for three-channel fluorescence microscopy."""

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int) -> int:
    """Choose the largest small GroupNorm divisor for a feature width."""
    return next(group for group in (8, 4, 2, 1) if channels % group == 0)


class ResidualBlock(nn.Module):
    """Two normalized convolutions with a learned residual projection."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.first = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
            nn.SiLU(inplace=True),
        )
        self.second = nn.Sequential(
            nn.Conv2d(output_channels, output_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(output_channels), output_channels),
        )
        self.projection = (
            nn.Identity()
            if input_channels == output_channels
            else nn.Conv2d(input_channels, output_channels, 1, bias=False)
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return normalized residual features with unchanged spatial size."""
        return self.activation(self.second(self.first(inputs)) + self.projection(inputs))


class ResidualUpBlock(nn.Module):
    """Bilinearly upsample, fuse an encoder skip, and refine residually."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.reduce = nn.Conv2d(input_channels, output_channels, 1, bias=False)
        self.refine = ResidualBlock(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Align decoder features exactly with the corresponding skip tensor."""
        inputs = F.interpolate(
            inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False
        )
        return self.refine(torch.cat((skip, self.reduce(inputs)), dim=1))


class MultiScaleContextBlock(nn.Module):
    """Fuse local and long-range context without a pretrained encoder.

    At the 1/16-resolution bottleneck, dilations one, two, and four cover cell
    bodies and long processes while keeping the parameter count practical.
    """

    def __init__(self, channels: int) -> None:
        super().__init__()
        hidden = max(8, channels // 4)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(channels, hidden, 1, bias=False),
                    nn.GroupNorm(_group_count(hidden), hidden),
                    nn.SiLU(inplace=True),
                    nn.Conv2d(
                        hidden,
                        hidden,
                        3,
                        padding=dilation,
                        dilation=dilation,
                        groups=hidden,
                        bias=False,
                    ),
                )
                for dilation in (1, 2, 4)
            ]
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(hidden * 3, channels, 1, bias=False),
            nn.GroupNorm(_group_count(channels), channels),
        )
        self.activation = nn.SiLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return a residual fusion of three bottleneck receptive fields."""
        context = self.fuse(torch.cat([branch(inputs) for branch in self.branches], dim=1))
        return self.activation(inputs + context)


class MultichannelInstanceUNet(nn.Module):
    """Four-level residual U-Net with compartment, boundary, and offset heads.

    GroupNorm is independent of the small microscopy batch size, while residual
    blocks and one additional resolution level provide a wider receptive field
    for thin processes than the compact historical baseline.
    """

    def __init__(
        self,
        input_channels: int = 3,
        compartment_classes: int = 4,
        base_channels: int = 24,
    ) -> None:
        super().__init__()
        if input_channels != 3:
            raise ValueError("MultichannelInstanceUNet requires exactly three input channels")
        if compartment_classes != 4:
            raise ValueError("The instance model requires four compartment classes")
        if base_channels <= 0:
            raise ValueError("base_channels must be positive")
        widths = [base_channels * factor for factor in (1, 2, 4, 8)]
        self.encoder1 = ResidualBlock(input_channels, widths[0])
        self.encoder2 = ResidualBlock(widths[0], widths[1])
        self.encoder3 = ResidualBlock(widths[1], widths[2])
        self.encoder4 = ResidualBlock(widths[2], widths[3])
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = nn.Sequential(
            ResidualBlock(widths[3], widths[3] * 2),
            nn.Dropout2d(0.1),
        )
        self.decoder4 = ResidualUpBlock(widths[3] * 2, widths[3], widths[3])
        self.decoder3 = ResidualUpBlock(widths[3], widths[2], widths[2])
        self.decoder2 = ResidualUpBlock(widths[2], widths[1], widths[1])
        self.decoder1 = ResidualUpBlock(widths[1], widths[0], widths[0])
        self.semantic_output = nn.Conv2d(widths[0], compartment_classes, 1)
        self.boundary_output = nn.Conv2d(widths[0], 2, 1)
        self.offset_output = nn.Conv2d(widths[0], 2, 1)

    def _decode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Produce full-resolution shared features from three fluorescence planes."""
        if inputs.ndim != 4 or inputs.shape[1] != 3:
            raise ValueError(
                f"MultichannelInstanceUNet expects [B, 3, H, W]; received {tuple(inputs.shape)}"
            )
        if min(inputs.shape[-2:]) < 16:
            raise ValueError("Input height and width must each be at least 16 pixels")
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        skip4 = self.encoder4(self.pool(skip3))
        decoded = self.bottleneck(self.pool(skip4))
        decoded = self.decoder4(decoded, skip4)
        decoded = self.decoder3(decoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        return self.decoder1(decoded, skip1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return all instance-training heads at the input spatial resolution."""
        features = self._decode_features(inputs)
        return {
            "semantic_logits": self.semantic_output(features),
            "boundary_logits": self.boundary_output(features),
            "offsets": self.offset_output(features),
        }


class AstroSegInstanceUNet(MultichannelInstanceUNet):
    """From-scratch AstroSeg network optimized for individual-cell masks.

    The legacy model is retained for old checkpoints. This version adds explicit
    binary foreground supervision and multiscale bottleneck context; all weights
    are initialized by PyTorch and learned only from the project dataset.
    """

    def __init__(
        self,
        input_channels: int = 3,
        compartment_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        super().__init__(input_channels, compartment_classes, base_channels)
        bottleneck_channels = base_channels * 16
        first_block = self.bottleneck[0]
        self.bottleneck = nn.Sequential(
            first_block,
            MultiScaleContextBlock(bottleneck_channels),
            nn.Dropout2d(0.15),
        )
        self.foreground_output = nn.Conv2d(base_channels, 2, 1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Predict foreground, compartments, boundaries, and nucleus offsets."""
        features = self._decode_features(inputs)
        return {
            "foreground_logits": self.foreground_output(features),
            "semantic_logits": self.semantic_output(features),
            "boundary_logits": self.boundary_output(features),
            "offsets": self.offset_output(features),
        }
