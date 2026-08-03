"""NaN-safe per-class segmentation metrics."""

from typing import Any

import numpy as np
import torch


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 1.0


def metrics_from_predictions(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    """Compute per-class and foreground-macro metrics from integer class masks."""
    predicted = predictions.detach().cpu().numpy() if isinstance(predictions, torch.Tensor) else np.asarray(predictions)
    target = targets.detach().cpu().numpy() if isinstance(targets, torch.Tensor) else np.asarray(targets)
    if predicted.shape != target.shape:
        raise ValueError(f"Prediction shape {predicted.shape} does not match target shape {target.shape}")
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2")
    if predicted.size and (
        predicted.min() < 0
        or target.min() < 0
        or predicted.max() >= num_classes
        or target.max() >= num_classes
    ):
        raise ValueError("Class masks contain values outside [0, num_classes)")

    result: dict[str, Any] = {"per_class": {}}
    names = ("dice", "iou", "precision", "recall")
    collected = {name: [] for name in names}
    for class_index in range(num_classes):
        pred_class = predicted == class_index
        target_class = target == class_index
        true_positive = float(np.logical_and(pred_class, target_class).sum())
        false_positive = float(np.logical_and(pred_class, ~target_class).sum())
        false_negative = float(np.logical_and(~pred_class, target_class).sum())
        values = {
            "dice": _safe_ratio(2.0 * true_positive, 2.0 * true_positive + false_positive + false_negative),
            "iou": _safe_ratio(true_positive, true_positive + false_positive + false_negative),
            "precision": _safe_ratio(true_positive, true_positive + false_positive),
            "recall": _safe_ratio(true_positive, true_positive + false_negative),
        }
        result["per_class"][class_index] = values
        if class_index > 0:
            for name in names:
                collected[name].append(values[name])
    result["macro"] = {name: float(np.mean(values)) for name, values in collected.items()}
    return result


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Convert logits to class masks and compute segmentation metrics."""
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, C, H, W]")
    return metrics_from_predictions(torch.argmax(logits, dim=1), targets, logits.shape[1])

