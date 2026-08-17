"""Object-level metrics for complete astrocyte instance segmentation."""

from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment


def _validate_instances(labels: np.ndarray, name: str) -> np.ndarray:
    """Validate a non-negative two-dimensional instance-label image."""
    array = np.asarray(labels)
    if array.ndim != 2 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty 2D array")
    if not np.issubdtype(array.dtype, np.number) or not np.isfinite(array).all():
        raise ValueError(f"{name} must contain finite numeric values")
    if np.any(array < 0) or not np.equal(array, np.floor(array)).all():
        raise ValueError(f"{name} must contain non-negative integer labels")
    return array.astype(np.int64, copy=False)


def match_instances(
    predicted: np.ndarray,
    target: np.ndarray,
) -> tuple[list[tuple[int, int, float]], np.ndarray]:
    """Find the one-to-one predicted/target assignment with maximum total IoU."""
    prediction = _validate_instances(predicted, "predicted")
    truth = _validate_instances(target, "target")
    if prediction.shape != truth.shape:
        raise ValueError("Predicted and target instance images must have identical dimensions")
    predicted_ids = np.unique(prediction[prediction > 0])
    target_ids = np.unique(truth[truth > 0])
    matrix = np.zeros((len(predicted_ids), len(target_ids)), dtype=np.float64)
    for row, predicted_id in enumerate(predicted_ids):
        predicted_mask = prediction == predicted_id
        overlapping_targets = np.unique(truth[predicted_mask])
        for target_id in overlapping_targets[overlapping_targets > 0]:
            column = int(np.searchsorted(target_ids, target_id))
            target_mask = truth == target_id
            intersection = np.logical_and(predicted_mask, target_mask).sum()
            union = np.logical_or(predicted_mask, target_mask).sum()
            matrix[row, column] = float(intersection / union)
    if matrix.size == 0:
        return [], matrix
    rows, columns = linear_sum_assignment(-matrix)
    matches = [
        (int(predicted_ids[row]), int(target_ids[column]), float(matrix[row, column]))
        for row, column in zip(rows, columns, strict=True)
    ]
    return matches, matrix


def instance_segmentation_metrics(
    predicted: np.ndarray,
    target: np.ndarray,
    iou_threshold: float = 0.5,
) -> dict[str, Any]:
    """Compute detection and overlap metrics for individual astrocytes."""
    if not 0 < iou_threshold <= 1:
        raise ValueError("iou_threshold must be in (0, 1]")
    prediction = _validate_instances(predicted, "predicted")
    truth = _validate_instances(target, "target")
    matches, _ = match_instances(prediction, truth)
    accepted = [match for match in matches if match[2] >= iou_threshold]
    predicted_count = int(np.unique(prediction[prediction > 0]).size)
    target_count = int(np.unique(truth[truth > 0]).size)
    true_positive = len(accepted)
    false_positive = predicted_count - true_positive
    false_negative = target_count - true_positive
    precision = true_positive / predicted_count if predicted_count else float(target_count == 0)
    recall = true_positive / target_count if target_count else float(predicted_count == 0)
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    mean_iou = float(np.mean([match[2] for match in accepted])) if accepted else 0.0
    denominator = true_positive + 0.5 * false_positive + 0.5 * false_negative
    panoptic_quality = (
        float(sum(match[2] for match in accepted) / denominator) if denominator else 1.0
    )
    return {
        "predicted_count": predicted_count,
        "target_count": target_count,
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "mean_matched_iou": mean_iou,
        "panoptic_quality": panoptic_quality,
        "matches": accepted,
    }


def process_ownership_accuracy(
    predicted: np.ndarray,
    target: np.ndarray,
    target_compartments: np.ndarray,
    process_class: int = 3,
) -> float:
    """Measure whether annotated process pixels retain the correct cell owner."""
    prediction = _validate_instances(predicted, "predicted")
    truth = _validate_instances(target, "target")
    compartments = np.asarray(target_compartments)
    if prediction.shape != truth.shape or compartments.shape != truth.shape:
        raise ValueError("Prediction, target, and compartment images must be aligned")
    process_pixels = (compartments == process_class) & (truth > 0)
    if not process_pixels.any():
        return 1.0
    matches, _ = match_instances(prediction, truth)
    target_to_prediction = {target_id: predicted_id for predicted_id, target_id, _ in matches}
    correct = np.zeros(truth.shape, dtype=bool)
    for target_id in np.unique(truth[process_pixels]):
        predicted_id = target_to_prediction.get(int(target_id), 0)
        correct |= process_pixels & (truth == target_id) & (prediction == predicted_id)
    return float(correct.sum() / process_pixels.sum())
