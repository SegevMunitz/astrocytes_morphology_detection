"""Compact two-dimensional U-Net baseline."""

import torch
from torch import nn
from torch.nn import functional as F


class DoubleConv(nn.Module):
    """Apply two padded convolution and ReLU transformations.

    Spatial dimensions remain unchanged, allowing outputs to participate in
    encoder-decoder skip connections without manual cropping.
    """

    def __init__(self, input_channels: int, output_channels: int) -> None:
        """Configure two spatially padded convolutions for one feature level.

        The first layer changes channel width and the second refines it, with an
        in-place ReLU following each convolution to keep the baseline compact.
        """
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_channels, output_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Transform a four-dimensional feature tensor through both convolutions.

        Batch and spatial dimensions are preserved; only the configured channel
        count changes. Shape validation is delegated to PyTorch convolutions.
        """
        return self.layers(inputs)


class UpBlock(nn.Module):
    """Decode one U-Net resolution level using an encoder skip tensor.

    Transposed convolution upsamples the decoder features, which are resized for
    odd dimensions, concatenated with the skip, and refined by ``DoubleConv``.
    """

    def __init__(self, input_channels: int, skip_channels: int, output_channels: int) -> None:
        """Configure learned upsampling and skip-feature refinement.

        Channel dimensions account for concatenating the upsampled decoder tensor
        with the matching encoder tensor before the final double convolution.
        """
        super().__init__()
        self.up = nn.ConvTranspose2d(input_channels, output_channels, kernel_size=2, stride=2)
        self.conv = DoubleConv(output_channels + skip_channels, output_channels)

    def forward(self, inputs: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        """Upsample decoder features and merge them with one encoder skip.

        Bilinear interpolation resolves one-pixel mismatches caused by pooling
        odd-sized inputs, ensuring that concatenation remains spatially safe.
        """
        inputs = self.up(inputs)
        if inputs.shape[-2:] != skip.shape[-2:]:
            inputs = F.interpolate(inputs, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.conv(torch.cat((skip, inputs), dim=1))


class UNet(nn.Module):
    """Compact two-dimensional U-Net for semantic microscopy segmentation.

    Three encoder levels and mirrored skip-connected decoder levels produce
    per-pixel logits. The network contains no fully connected classification head.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 2, base_channels: int = 32) -> None:
        """Build the complete encoder, bottleneck, decoder, and logit projection.

        Input channels, output classes, and base feature width are configurable;
        invalid non-positive widths or fewer than two classes are rejected.
        """
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

    def _decode_features(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and decode a batch into full-resolution shared features.

        Inputs must follow ``[B, C, H, W]`` and be at least eight pixels per
        spatial axis. The returned feature map supports one or several task heads.
        """
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
        return self.decoder1(decoded, skip1)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """Encode and decode a batch into full-resolution class logits.

        The semantic baseline retains its tensor-only API while the shared feature
        method allows the instance model to add boundary and ownership heads.
        """
        return self.output(self._decode_features(inputs))


class NucleusGuidedInstanceUNet(UNet):
    """Multi-head U-Net for complete nucleus-owned astrocyte instances.

    The semantic head distinguishes background, nucleus, soma, and process. A
    boundary head separates touching cells, while offsets point every cell pixel
    toward the nucleus that owns it.
    """

    def __init__(
        self,
        input_channels: int = 3,
        compartment_classes: int = 4,
        base_channels: int = 32,
    ) -> None:
        """Build a shared U-Net decoder with three full-resolution task heads.

        Four compartment classes are the supported scientific contract. Boundary
        logits are binary and offsets contain scale-normalized ``dy, dx`` values.
        """
        if compartment_classes != 4:
            raise ValueError("The instance model requires four compartment classes")
        super().__init__(input_channels, compartment_classes, base_channels)
        self.boundary_output = nn.Conv2d(base_channels, 2, kernel_size=1)
        self.offset_output = nn.Conv2d(base_channels, 2, kernel_size=1)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        """Return semantic logits, boundary logits, and ownership offsets.

        The regression head remains unbounded so processes longer than the target
        normalization scale still reach their owning nucleus without truncation.
        """
        features = self._decode_features(inputs)
        return {
            "semantic_logits": self.output(features),
            "boundary_logits": self.boundary_output(features),
            "offsets": self.offset_output(features),
        }
