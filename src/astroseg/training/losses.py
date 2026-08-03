"""Differentiable segmentation losses."""

import torch
from torch import nn
from torch.nn import functional as F


class DiceLoss(nn.Module):
    """Soft multiclass Dice loss with optional background exclusion."""

    def __init__(self, include_background: bool = False, smooth: float = 1e-6) -> None:
        super().__init__()
        if smooth <= 0:
            raise ValueError("smooth must be positive")
        self.include_background = include_background
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar mean Dice loss across batch and selected classes."""
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
    """Weighted sum of cross-entropy and soft Dice losses."""

    def __init__(
        self,
        cross_entropy_weight: float = 1.0,
        dice_weight: float = 1.0,
        include_background: bool = False,
    ) -> None:
        super().__init__()
        if cross_entropy_weight < 0 or dice_weight < 0 or cross_entropy_weight + dice_weight == 0:
            raise ValueError("Loss weights must be non-negative and at least one must be positive")
        self.cross_entropy_weight = cross_entropy_weight
        self.dice_weight = dice_weight
        self.dice = DiceLoss(include_background=include_background)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Return scalar weighted cross-entropy plus Dice loss."""
        cross_entropy = F.cross_entropy(logits, target.long())
        return self.cross_entropy_weight * cross_entropy + self.dice_weight * self.dice(logits, target)

