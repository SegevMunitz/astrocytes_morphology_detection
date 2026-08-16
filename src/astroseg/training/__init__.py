"""Losses, metrics, checkpoints, and training loops."""

from astroseg.training.losses import CrossEntropyDiceLoss, DiceLoss, NucleusGuidedInstanceLoss
from astroseg.training.instance_metrics import (
    instance_segmentation_metrics,
    match_instances,
    process_ownership_accuracy,
)
from astroseg.training.instance_trainer import (
    InstanceEpochMetrics,
    run_instance_epoch,
    run_instance_overfit_smoke_test,
    train_instance_model,
)
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
    "InstanceEpochMetrics",
    "NucleusGuidedInstanceLoss",
    "instance_segmentation_metrics",
    "match_instances",
    "process_ownership_accuracy",
    "run_instance_epoch",
    "run_instance_overfit_smoke_test",
    "assign_grouped_folds",
    "load_grouped_fold_manifests",
    "metrics_from_logits",
    "metrics_from_probability_patches",
    "metrics_from_predictions",
    "run_overfit_smoke_test",
    "set_deterministic_seed",
    "split_grouped_fold",
    "train_model",
    "train_instance_model",
]
