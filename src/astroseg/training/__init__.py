"""Losses, metrics, checkpoints, and training loops."""

from astroseg.training.losses import CrossEntropyDiceLoss, DiceLoss
from astroseg.training.metrics import (
    metrics_from_logits,
    metrics_from_predictions,
    metrics_from_probability_patches,
)
from astroseg.training.trainer import run_overfit_smoke_test, set_deterministic_seed, train_model
from astroseg.training.cross_validation import (
    assign_grouped_folds,
    load_grouped_fold_manifests,
    split_grouped_fold,
)

__all__ = [
    "CrossEntropyDiceLoss",
    "DiceLoss",
    "assign_grouped_folds",
    "load_grouped_fold_manifests",
    "metrics_from_logits",
    "metrics_from_probability_patches",
    "metrics_from_predictions",
    "run_overfit_smoke_test",
    "set_deterministic_seed",
    "split_grouped_fold",
    "train_model",
]
