"""Train the nucleus-guided model that separates complete individual astrocytes."""

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import torch
from torch.utils.data import DataLoader

from astroseg.constants import TRAINABLE_ANNOTATION_STATUSES
from astroseg.datasets import (
    AstrocyteInstanceDataset,
    AstrocyteUnlabeledDataset,
    RandomInstanceAugmentation,
    collate_instance_batch,
    collate_unlabeled_batch,
)
from astroseg.io import load_manifest, load_yaml_configuration
from astroseg.models import build_model
from astroseg.training import (
    NucleusGuidedInstanceLoss,
    assign_grouped_folds,
    run_instance_overfit_smoke_test,
    set_deterministic_seed,
    split_grouped_fold,
    train_instance_model,
)


def load_configuration(path: Path) -> dict[str, Any]:
    """Load the instance-training YAML and check its required top-level sections.

    Detailed values are validated by datasets, model, loss, and trainer constructors,
    keeping the YAML loader focused on structural errors and readable messages.
    """
    return load_yaml_configuration(
        path, required_sections=("data", "model", "training", "loss", "output")
    )


def _eligible_instance_rows(
    manifest_path: Path, statuses: set[str]
) -> pd.DataFrame:
    """Select human-trainable rows that contain full-cell instance supervision.

    Binary-only and automatic pseudo annotations are intentionally excluded so
    they cannot silently train ownership offsets as if they were ground truth.
    """
    manifest = load_manifest(manifest_path)
    selected = manifest["annotation_status"].isin(statuses)
    selected &= manifest["instance_annotation_path"].astype(str).str.strip() != ""
    selected &= manifest["split"] != "test"
    result = manifest.loc[selected].copy()
    if result.empty:
        raise ValueError(
            "No trainable full-cell instance annotations were found. Import corrected "
            "instance masks before running real instance training."
        )
    return result


def _read_image_ids(path: Path) -> set[str]:
    """Read a non-empty comment-aware image-ID list for an explicit split."""
    values = {
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    if not values:
        raise ValueError(f"No image IDs in {path}")
    return values


def train_from_configuration(configuration: dict[str, Any]) -> list[dict[str, float | int]]:
    """Build grouped data loaders and train the complete-cell multi-head U-Net.

    Fold assignment happens before patch extraction, guaranteeing that patches from
    one image or well never occur in both training and validation.
    """
    seed = int(configuration.get("seed", 42))
    set_deterministic_seed(seed)
    data = configuration["data"]
    model_configuration = configuration["model"]
    manifest_path = Path(data["manifest_path"])
    statuses = {
        str(value).strip().lower()
        for value in data.get("train_annotation_statuses", TRAINABLE_ANNOTATION_STATUSES)
    }
    candidates = _eligible_instance_rows(manifest_path, statuses)
    cross_validation = configuration.get("cross_validation", {})
    output_directory = Path(configuration["output"]["directory"])
    explicit_split = configuration.get("explicit_split", {})
    full_training = configuration.get("full_training", {})
    if full_training.get("enabled", False):
        if explicit_split.get("enabled", False):
            raise ValueError("Full-data training cannot use an explicit validation split")
        train_manifest = candidates.copy()
        train_manifest["split"] = "train"
        validation_manifest = None
        output_directory.mkdir(parents=True, exist_ok=True)
        train_manifest.to_csv(
            output_directory / "cross_validation_assignments.csv", index=False
        )
    elif explicit_split.get("enabled", False):
        training_ids = _read_image_ids(Path(explicit_split["train_ids_path"]))
        validation_ids = _read_image_ids(Path(explicit_split["validation_ids_path"]))
        overlap = training_ids & validation_ids
        if overlap:
            raise ValueError(f"Explicit train/validation lists overlap: {sorted(overlap)}")
        candidate_ids = set(candidates["image_id"].astype(str))
        listed_ids = training_ids | validation_ids
        if listed_ids != candidate_ids:
            raise ValueError(
                "Explicit split must cover every labeled candidate exactly; "
                f"missing={sorted(candidate_ids - listed_ids)}, "
                f"unknown={sorted(listed_ids - candidate_ids)}"
            )
        train_manifest = candidates.loc[candidates["image_id"].isin(training_ids)].copy()
        validation_manifest = candidates.loc[
            candidates["image_id"].isin(validation_ids)
        ].copy()
        train_manifest["split"] = "train"
        validation_manifest["split"] = "val"
        output_directory.mkdir(parents=True, exist_ok=True)
        pd.concat((train_manifest, validation_manifest), ignore_index=True).to_csv(
            output_directory / "cross_validation_assignments.csv", index=False
        )
    elif cross_validation.get("enabled", True):
        group_column = str(cross_validation.get("group_column", "image_id"))
        fold_column = str(cross_validation.get("fold_column", "fold"))
        n_splits = int(cross_validation.get("n_splits", 5))
        if fold_column not in candidates.columns or (
            candidates[fold_column].astype(str).str.strip() == ""
        ).any():
            candidates = assign_grouped_folds(
                candidates, n_splits, group_column, fold_column, seed
            )
        train_manifest, validation_manifest = split_grouped_fold(
            candidates,
            int(cross_validation.get("validation_fold", 0)),
            group_column,
            fold_column,
        )
        output_directory.mkdir(parents=True, exist_ok=True)
        pd.concat((train_manifest, validation_manifest), ignore_index=True).to_csv(
            output_directory / "cross_validation_assignments.csv", index=False
        )
    else:
        train_manifest = candidates.loc[candidates["split"] == "train"].copy()
        validation_manifest = candidates.loc[candidates["split"] == "val"].copy()
        if train_manifest.empty or validation_manifest.empty:
            raise ValueError("Non-CV training requires eligible train and val manifest rows")

    common = {
        "patch_size": int(data["patch_size"]),
        "overlap": int(data["overlap"]),
        "max_nucleus_distance": float(data.get("max_nucleus_distance", 64)),
        "soma_radius": float(data.get("soma_radius", 20)),
        "offset_scale": float(data.get("offset_scale", 256)),
        "annotation_statuses": statuses,
        "manifest_base_directory": manifest_path.parent,
        "input_mode": str(data.get("input_mode", "nucleus_guidance")),
    }
    augmentation = configuration.get("augmentation", {})
    train_dataset = AstrocyteInstanceDataset(
        train_manifest,
        "train",
        augmentation=RandomInstanceAugmentation(
            auxiliary_dropout_probability=float(
                augmentation.get("auxiliary_dropout_probability", 0)
            )
        ),
        **common,
    )
    validation_dataset = (
        AstrocyteInstanceDataset(validation_manifest, "val", augmentation=None, **common)
        if validation_manifest is not None
        else None
    )
    loader_options = {
        "batch_size": int(configuration["training"]["batch_size"]),
        "num_workers": int(data.get("num_workers", 0)),
        "collate_fn": collate_instance_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = (
        DataLoader(validation_dataset, shuffle=False, **loader_options)
        if validation_dataset is not None
        else None
    )
    unlabeled_loader = None
    semi_supervised = configuration.get("semi_supervised", {})
    if semi_supervised.get("enabled", False):
        complete_manifest = load_manifest(manifest_path)
        unlabeled_dataset = AstrocyteUnlabeledDataset(
            complete_manifest,
            patch_size=int(data["patch_size"]),
            overlap=int(data["overlap"]),
            max_nucleus_distance=float(data.get("max_nucleus_distance", 64)),
            manifest_base_directory=manifest_path.parent,
            input_mode=str(data.get("input_mode", "fluorescence")),
        )
        unlabeled_loader = DataLoader(
            unlabeled_dataset,
            batch_size=int(configuration["training"]["batch_size"]),
            shuffle=True,
            num_workers=int(data.get("num_workers", 0)),
            collate_fn=collate_unlabeled_batch,
        )
    model = build_model(
        str(model_configuration["architecture"]),
        int(model_configuration["input_channels"]),
        int(model_configuration["num_classes"]),
        int(model_configuration.get("base_channels", 32)),
    )
    initialization = str(model_configuration.get("initialization", "random")).lower()
    if initialization != "random":
        raise ValueError(
            "Custom instance training supports random initialization only; "
            "pretrained model weights must not enter this pipeline"
        )
    loss = configuration["loss"]
    criterion = NucleusGuidedInstanceLoss(
        semantic_weight=float(loss.get("semantic_weight", 1)),
        boundary_weight=float(loss.get("boundary_weight", 1)),
        offset_weight=float(loss.get("offset_weight", 1)),
        foreground_weight=float(loss.get("foreground_weight", 0)),
        semantic_class_weights=loss.get("semantic_class_weights"),
        boundary_class_weights=loss.get("boundary_class_weights"),
        foreground_class_weights=loss.get("foreground_class_weights"),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    summary = {
        "architecture": str(model_configuration["architecture"]),
        "initialization": initialization,
        "parameter_count": int(sum(parameter.numel() for parameter in model.parameters())),
        "labeled_training_images": int(train_manifest["image_id"].nunique()),
        "validation_images": (
            int(validation_manifest["image_id"].nunique())
            if validation_manifest is not None
            else 0
        ),
        "supervised_training_patches": len(train_dataset),
        "validation_patches": len(validation_dataset) if validation_dataset is not None else 0,
        "unlabeled_training_images": (
            int(unlabeled_loader.dataset.manifest["image_id"].nunique())
            if unlabeled_loader is not None
            else 0
        ),
        "unlabeled_training_patches": (
            len(unlabeled_loader.dataset) if unlabeled_loader is not None else 0
        ),
    }
    (output_directory / "training_data_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return train_instance_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        configuration,
        output_directory,
        device,
        unlabeled_loader=unlabeled_loader,
    )


def parse_args() -> argparse.Namespace:
    """Parse configured training, fold override, and synthetic smoke-test options.

    Smoke mode needs no annotation files. Real training always requires the
    instance-specific configuration and full-cell corrected masks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--train-id-list", type=Path)
    parser.add_argument("--validation-id-list", type=Path)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-all", action="store_true")
    parser.add_argument("--disable-lr-scheduler", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run either the engineering smoke test or a complete configured training run.

    The command refuses real training when instance annotations are missing rather
    than falling back to binary masks that lack process ownership information.
    """
    args = parse_args()
    if args.smoke_test:
        initial, final = run_instance_overfit_smoke_test(args.smoke_steps)
        print(f"Instance smoke test passed: loss {initial:.6f} -> {final:.6f}")
        return
    if args.config is None:
        raise SystemExit("--config is required unless --smoke-test is used")
    configuration = load_configuration(args.config)
    if args.train_all:
        configuration["full_training"] = {"enabled": True}
        configuration.setdefault("cross_validation", {})["enabled"] = False
        configuration["training"]["checkpoint_metric"] = "training_instance_proxy"
        configuration["training"]["lr_scheduler"] = "cosine"
    if args.seed is not None:
        if args.seed < 0:
            raise SystemExit("--seed must be non-negative")
        configuration["seed"] = args.seed
    if (args.train_id_list is None) != (args.validation_id_list is None):
        raise SystemExit("--train-id-list and --validation-id-list must be supplied together")
    if args.train_id_list is not None:
        if args.fold is not None:
            raise SystemExit("An explicit train/validation split cannot be combined with --fold")
        configuration["explicit_split"] = {
            "enabled": True,
            "train_ids_path": str(args.train_id_list),
            "validation_ids_path": str(args.validation_id_list),
        }
    if args.fold is not None:
        configuration.setdefault("cross_validation", {})["enabled"] = True
        configuration["cross_validation"]["validation_fold"] = args.fold
    if args.epochs is not None:
        if args.epochs <= 0:
            raise SystemExit("--epochs must be positive")
        configuration["training"]["epochs"] = args.epochs
    if args.learning_rate is not None:
        if args.learning_rate <= 0:
            raise SystemExit("--learning-rate must be positive")
        configuration["training"]["learning_rate"] = args.learning_rate
    if args.disable_lr_scheduler:
        configuration["training"]["lr_scheduler_enabled"] = False
    if args.output_dir is not None:
        configuration["output"]["directory"] = str(args.output_dir)
    history = train_from_configuration(configuration)
    print(f"Completed {len(history)} instance-training epoch(s)")


if __name__ == "__main__":
    main()
