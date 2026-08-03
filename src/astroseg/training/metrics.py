"""NaN-safe per-class segmentation metrics."""

from typing import Any

import numpy as np
import torch

from astroseg.preprocessing.patches import PatchCoordinates, stitch_probability_patches


def _safe_overlap_ratio(numerator: float, denominator: float) -> float:
    """Return an overlap ratio with perfect score for two empty supports.

    Dice and IoU denominators reach zero only when prediction and target both lack
    the class, which is treated as agreement rather than NaN.
    """
    return numerator / denominator if denominator > 0 else 1.0


def metrics_from_predictions(
    predictions: np.ndarray | torch.Tensor,
    targets: np.ndarray | torch.Tensor,
    num_classes: int,
) -> dict[str, Any]:
    """Compute per-class metrics from aligned integer prediction and target masks.

    Dice, IoU, precision, and recall are returned for every class, plus a macro
    average over foreground classes. Empty denominators use explicit finite rules.
    """
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
        predicted_positive = true_positive + false_positive
        target_positive = true_positive + false_negative
        values = {
            "dice": _safe_overlap_ratio(
                2.0 * true_positive,
                2.0 * true_positive + false_positive + false_negative,
            ),
            "iou": _safe_overlap_ratio(
                true_positive, true_positive + false_positive + false_negative
            ),
            "precision": (
                true_positive / predicted_positive
                if predicted_positive > 0
                else float(target_positive == 0)
            ),
            "recall": (
                true_positive / target_positive
                if target_positive > 0
                else float(predicted_positive == 0)
            ),
        }
        result["per_class"][class_index] = values
        if class_index > 0:
            for name in names:
                collected[name].append(values[name])
    result["macro"] = {name: float(np.mean(values)) for name, values in collected.items()}
    return result


def metrics_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, Any]:
    """Convert batched logits to hard class masks and compute standard metrics.

    Argmax is applied along the class channel before delegating to the mask-based
    metric API. This function is intended for batch monitoring during training.
    """
    if logits.ndim != 4:
        raise ValueError("logits must have shape [B, C, H, W]")
    return metrics_from_predictions(torch.argmax(logits, dim=1), targets, logits.shape[1])


def metrics_from_probability_patches(
    probability_patches: list[np.ndarray],
    target_patches: list[np.ndarray],
    coordinates: list[PatchCoordinates],
    image_shape: tuple[int, int],
) -> dict[str, Any]:
    """Reconstruct one full image before computing segmentation metrics.

    Probability patches are averaged in overlaps, while target patches must agree
    exactly wherever they overlap. This prevents overlap pixels from receiving
    extra weight merely because they occur in several evaluation patches.
    """
    if not probability_patches or len(probability_patches) != len(target_patches):
        raise ValueError("Probability and target patch lists must have the same non-zero length")
    if len(coordinates) != len(probability_patches):
        raise ValueError("Patch coordinates must match the number of prediction patches")
    num_classes = probability_patches[0].shape[0]
    full_probabilities = stitch_probability_patches(
        probability_patches, coordinates, image_shape
    )
    full_target = np.full(image_shape, -1, dtype=np.int64)
    for target_patch, coordinate in zip(target_patches, coordinates, strict=True):
        target_array = np.asarray(target_patch)
        expected_shape = (coordinate.height, coordinate.width)
        if target_array.shape != expected_shape:
            raise ValueError(
                f"Target patch shape {target_array.shape} does not match coordinates {expected_shape}"
            )
        y2 = coordinate.y + coordinate.height
        x2 = coordinate.x + coordinate.width
        current = full_target[coordinate.y:y2, coordinate.x:x2]
        overlap = current >= 0
        if np.any(current[overlap] != target_array[overlap]):
            raise ValueError("Overlapping target patches contain inconsistent class labels")
        current[~overlap] = target_array[~overlap]
    if np.any(full_target < 0):
        raise ValueError("Target patches do not cover the complete image")
    return metrics_from_predictions(
        full_probabilities.argmax(axis=0), full_target, num_classes
    )
