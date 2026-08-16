"""Differentiable segmentation losses."""

import torch
from torch import nn
from torch.nn import functional as F
from collections.abc import Sequence


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
        class_weights: Sequence[float] | None = None,
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
        weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        if weights is not None and (weights.ndim != 1 or torch.any(weights <= 0)):
            raise ValueError("class_weights must be a one-dimensional positive sequence")
        self.register_buffer("class_weights", weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute the configured weighted loss for one prediction batch.

        Logits and integer class targets are passed to both component losses, and
        their weighted scalar values are summed for optimization.
        """
        if self.class_weights is not None and len(self.class_weights) != logits.shape[1]:
            raise ValueError("class_weights length must match the number of logit classes")
        cross_entropy = F.cross_entropy(logits, target.long(), weight=self.class_weights)
        return self.cross_entropy_weight * cross_entropy + self.dice_weight * self.dice(logits, target)


class NucleusGuidedInstanceLoss(nn.Module):
    """Joint compartment, cell-boundary, and nucleus-ownership objective.

    Semantic and boundary heads use cross-entropy plus Dice. Smooth-L1 offset loss
    is evaluated only on pixels belonging to an annotated astrocyte instance.
    """

    def __init__(
        self,
        semantic_weight: float = 1.0,
        boundary_weight: float = 1.0,
        offset_weight: float = 1.0,
        semantic_class_weights: Sequence[float] | None = None,
        boundary_class_weights: Sequence[float] | None = None,
        foreground_weight: float = 0.0,
        foreground_class_weights: Sequence[float] | None = None,
    ) -> None:
        """Configure non-negative task weights with at least one active objective.

        Separate weights allow boundary and long-process ownership supervision to
        be balanced against the more numerous semantic pixels.
        """
        super().__init__()
        weights = (semantic_weight, boundary_weight, offset_weight, foreground_weight)
        if any(weight < 0 for weight in weights) or sum(weights) == 0:
            raise ValueError("Instance-loss weights must be non-negative with a positive sum")
        self.semantic_weight = semantic_weight
        self.boundary_weight = boundary_weight
        self.offset_weight = offset_weight
        self.foreground_weight = foreground_weight
        self.semantic_loss = CrossEntropyDiceLoss(class_weights=semantic_class_weights)
        self.boundary_loss = CrossEntropyDiceLoss(class_weights=boundary_class_weights)
        self.foreground_loss = CrossEntropyDiceLoss(
            class_weights=foreground_class_weights
        )

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute the weighted multi-head loss for one aligned training batch.

        Expected keys are explicit so an ordinary semantic model or incomplete
        target dictionary cannot be trained accidentally under the instance task.
        """
        return self.components(outputs, targets)["total"]

    def components(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Return weighted total and unweighted per-head losses for diagnostics."""
        output_keys = {"semantic_logits", "boundary_logits", "offsets"}
        if self.foreground_weight > 0:
            output_keys.add("foreground_logits")
        target_keys = {"semantic", "boundary", "offsets", "offset_mask"}
        missing_outputs = output_keys - set(outputs)
        missing_targets = target_keys - set(targets)
        if missing_outputs or missing_targets:
            raise ValueError(
                f"Incomplete instance batch; missing outputs={sorted(missing_outputs)}, "
                f"targets={sorted(missing_targets)}"
            )
        semantic = self.semantic_loss(outputs["semantic_logits"], targets["semantic"].long())
        boundary = self.boundary_loss(outputs["boundary_logits"], targets["boundary"].long())
        foreground = torch.zeros((), device=outputs["semantic_logits"].device)
        if self.foreground_weight > 0:
            foreground_target = (targets["semantic"] > 0).long()
            foreground = self.foreground_loss(
                outputs["foreground_logits"], foreground_target
            )
        predicted_offsets = outputs["offsets"]
        target_offsets = targets["offsets"].to(predicted_offsets.dtype)
        offset_mask = targets["offset_mask"].to(predicted_offsets.dtype)
        if predicted_offsets.shape != target_offsets.shape:
            raise ValueError("Predicted and target offsets must have identical shapes")
        if offset_mask.ndim != 3 or offset_mask.shape != predicted_offsets.shape[:1] + predicted_offsets.shape[2:]:
            raise ValueError("offset_mask must have shape [B, H, W]")
        per_value = F.smooth_l1_loss(predicted_offsets, target_offsets, reduction="none")
        expanded_mask = offset_mask.unsqueeze(1).expand_as(per_value)
        denominator = expanded_mask.sum().clamp_min(1.0)
        offset = (per_value * expanded_mask).sum() / denominator
        total = (
            self.semantic_weight * semantic
            + self.boundary_weight * boundary
            + self.offset_weight * offset
            + self.foreground_weight * foreground
        )
        return {
            "total": total,
            "semantic": semantic,
            "boundary": boundary,
            "offset": offset,
            "foreground": foreground,
        }
