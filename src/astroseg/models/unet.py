"""Compact two-dimensional U-Net baseline."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Two padded convolution and ReLU blocks."""

    def __init__(self, input_channels: int, output_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Apply the convolution block."""
        return self.layers(inputs)


class UpBlock(nn.Module):
    """Upsample, concatenate an encoder skip, and refine."""

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Decode one level, interpolating safely for odd input dimensions."""
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet(nn.Module):
    """A compact U-Net that returns per-pixel class logits."""

    def __init__(self, input_channels: int = 3, num_classes: int = 2, base_channels: int = 32) -> None:
        super().__init__()
        if input_channels <= 0 or num_classes < 2 or base_channels <= 0:
            raise ValueError("input_channels and base_channels must be positive; num_classes must be >= 2")
        self.encoder1 = DoubleConv(input_channels, base_channels)
        self.encoder2 = DoubleConv(base_channels, base_channels * 2)
        self.encoder3 = DoubleConv(base_channels * 2, base_channels * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base_channels * 4, base_channels * 8)
        self.decoder3 = UpBlock(base_channels * 8, base_channels * 4, base_channels * 4)
        self.decoder2 = UpBlock(base_channels * 4, base_channels * 2, base_channels * 2)
        self.decoder1 = UpBlock(base_channels * 2, base_channels, base_channels)
        self.output = nn.Conv2d(base_channels, num_classes, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Return logits with shape ``[B, num_classes, H, W]``."""
        if inputs.ndim != 4:
            raise ValueError(f"UNet expects [B, C, H, W] input; received {tuple(inputs.shape)}")
        if min(inputs.shape[-2:]) < 8:
            raise ValueError("UNet input height and width must each be at least 8 pixels")
        skip1 = self.encoder1(inputs)
        skip2 = self.encoder2(self.pool(skip1))
        skip3 = self.encoder3(self.pool(skip2))
        encoded = self.bottleneck(self.pool(skip3))
        decoded = self.decoder3(encoded, skip3)
        decoded = self.decoder2(decoded, skip2)
        decoded = self.decoder1(decoded, skip1)
        return self.output(decoded)

