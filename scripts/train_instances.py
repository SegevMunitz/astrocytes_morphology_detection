"""Train the nucleus-guided model that separates complete individual astrocytes."""

import argparse
from pathlib import Path
from typing import Any

import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from astroseg.constants import TRAINABLE_ANNOTATION_STATUSES
from astroseg.datasets import AstrocyteInstanceDataset, RandomInstanceFlip, collate_instance_batch
from astroseg.io import load_manifest
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
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Instance training configuration must be a YAML mapping")
    for section in ("data", "model", "training", "loss", "output"):
        if not isinstance(value.get(section), dict):
            raise ValueError(f"Configuration is missing mapping section {section!r}")
    return value


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
    if cross_validation.get("enabled", True):
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
    }
    train_dataset = AstrocyteInstanceDataset(
        train_manifest, "train", augmentation=RandomInstanceFlip(), **common
    )
    validation_dataset = AstrocyteInstanceDataset(
        validation_manifest, "val", augmentation=None, **common
    )
    loader_options = {
        "batch_size": int(configuration["training"]["batch_size"]),
        "num_workers": int(data.get("num_workers", 0)),
        "collate_fn": collate_instance_batch,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    validation_loader = DataLoader(validation_dataset, shuffle=False, **loader_options)
    model = build_model(
        str(model_configuration["architecture"]),
        int(model_configuration["input_channels"]),
        int(model_configuration["num_classes"]),
        int(model_configuration.get("base_channels", 32)),
    )
    loss = configuration["loss"]
    criterion = NucleusGuidedInstanceLoss(
        float(loss.get("semantic_weight", 1)),
        float(loss.get("boundary_weight", 1)),
        float(loss.get("offset_weight", 1)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_instance_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        configuration,
        output_directory,
        device,
    )


def parse_args() -> argparse.Namespace:
    """Parse configured training, fold override, and synthetic smoke-test options.

    Smoke mode needs no annotation files. Real training always requires the
    instance-specific configuration and full-cell corrected masks.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--epochs", type=int)
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
    if args.fold is not None:
        configuration.setdefault("cross_validation", {})["enabled"] = True
        configuration["cross_validation"]["validation_fold"] = args.fold
    if args.epochs is not None:
        if args.epochs <= 0:
            raise SystemExit("--epochs must be positive")
        configuration["training"]["epochs"] = args.epochs
    if args.output_dir is not None:
        configuration["output"]["directory"] = str(args.output_dir)
    history = train_from_configuration(configuration)
    print(f"Completed {len(history)} instance-training epoch(s)")


if __name__ == "__main__":
    main()
