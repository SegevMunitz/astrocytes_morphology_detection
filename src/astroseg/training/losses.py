"""Differentiable segmentation losses."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Differentiable multiclass Dice loss computed from softmax probabilities.

    Dice is aggregated across batch and spatial dimensions for each class, with
    optional background exclusion and smoothing for numerically safe empty classes.
    """

    def __init__(self, include_background: bool = False, smooth: float = 1e-6) -> None:
        """Configure background inclusion and numerical Dice smoothing.

        Positive smoothing prevents division by zero for empty classes, while the
        background flag controls whether class zero contributes to mean loss.
        """
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compare per-pixel logits with integer targets and return scalar loss.

        Targets are converted to one-hot planes, and classwise soft Dice values
        are averaged after applying the configured background policy.
        """
        if logits.ndim != 4 or target.ndim != 3 or logits.shape[0] != target.shape[0]:
            raise ValueError("Expected logits [B, C, H, W] and target [B, H, W]")
        num_classes = logits.shape[1]
        if num_classes < 2:
            raise ValueError("DiceLoss requires two or more classes")
        if target.shape[-2:] != logits.shape[-2:]:
            raise ValueError("Logit and target spatial shapes must match")
        if target.numel() and (target.min() < 0 or target.max() >= num_classes):
            raise ValueError("target contains class indices outside the logit channel range")
        probabilities = torch.softmax(logits, dim=1)
        one_hot = F.one_hot(target.long(), num_classes).permute(0, 3, 1, 2).to(probabilities.dtype)
        dims = (0, 2, 3)
        intersection = torch.sum(probabilities * one_hot, dim=dims)
        denominator = torch.sum(probabilities + one_hot, dim=dims)
        dice = (2.0 * intersection + self.smooth) / (denominator + self.smooth)
        if not self.include_background:
            dice = dice[1:]
        return 1.0 - dice.mean()


class CrossEntropyDiceLoss(nn.Module):
    """Weighted combination of categorical cross-entropy and soft Dice loss.

    Cross-entropy supplies pixelwise classification pressure while Dice emphasizes
    region overlap. Either component may be disabled by assigning it zero weight.
    """

    def __init__(
        self,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = False,
    ) -> None:
        """Configure component weights and the internal Dice-loss policy.

        Weights must be non-negative and cannot both be zero, ensuring that the
        resulting scalar always supplies an optimization objective.
        """
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0 or cross_entropy_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and at least one must be positive")
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(include_background=include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the configured weighted loss for one prediction batch.

        Logits and integer class targets are passed to both component losses, and
        their weighted scalar values are summed for optimization.
        """
        cross_entropy = F.cross_entropy(logits, target.long())
        return self.cross_entropy_weight * cross_entropy + self.dice_weight * self.dice(logits, target)
