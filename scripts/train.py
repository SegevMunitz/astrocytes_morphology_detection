"""Train the configured U-Net baseline or run a CPU overfit diagnostic."""

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from torch.utils.data import DataLoader

from astroseg.datasets import AstrocyteDataset, RandomFlip, collate_segmentation_batch
from astroseg.models import build_model
from astroseg.constants import TRAINABLE_ANNOTATION_STATUSES
from astroseg.training import (
    CrossEntropyDiceLoss,
    load_grouped_fold_manifests,
    run_overfit_smoke_test,
    set_deterministic_seed,
    train_model,
)


def load_configuration(path: Path) -> dict[str, Any]:
    """Load a YAML mapping used for model training."""
    if not path.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {path}")
    with path.open("r", encoding="utf-8") as handle:
        configuration = yaml.safe_load(handle)
    if not isinstance(configuration, dict):
        raise ValueError("Training configuration must be a YAML mapping")
    for section in ("data", "model", "training", "loss", "output"):
        if section not in configuration or not isinstance(configuration[section], dict):
            raise ValueError(f"Training configuration is missing mapping section {section!r}")
    return configuration


def train_from_configuration(configuration: dict[str, Any]) -> list[dict[str, float | int]]:
    """Construct datasets, loaders, model, loss, optimizer loop, and checkpoints."""
    set_deterministic_seed(int(configuration.get("seed", 42)))
    data_config = configuration["data"]
    manifest_path = Path(data_config["manifest_path"])
    annotation_statuses = data_config.get(
        "train_annotation_statuses", sorted(TRAINABLE_ANNOTATION_STATUSES)
    )
    common = {
        "patch_size": int(data_config["patch_size"]),
        "overlap": int(data_config["overlap"]),
        "max_nucleus_distance": float(data_config.get("max_nucleus_distance", 64.0)),
        "annotation_statuses": annotation_statuses,
    }
    cross_validation = configuration.get("cross_validation", {})
    if cross_validation.get("enabled", False):
        train_manifest, validation_manifest = load_grouped_fold_manifests(
            manifest_path,
            int(cross_validation.get("n_splits", 5)),
            int(cross_validation.get("validation_fold", 0)),
            str(cross_validation.get("group_column", "image_id")),
            str(cross_validation.get("fold_column", "fold")),
            int(configuration.get("seed", 42)),
            annotation_statuses,
        )
        frame_common = {"manifest_base_directory": manifest_path.parent}
        train_dataset = AstrocyteDataset(
            train_manifest, split="train", augmentation=RandomFlip(), **common, **frame_common
        )
        validation_dataset = AstrocyteDataset(
            validation_manifest, split="val", augmentation=None, **common, **frame_common
        )
    else:
        train_dataset = AstrocyteDataset(
            manifest_path, split="train", augmentation=RandomFlip(), **common
        )
        validation_dataset = AstrocyteDataset(
            manifest_path, split="val", augmentation=None, **common
        )
    batch_size = int(configuration["training"]["batch_size"])
    num_workers = int(data_config.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_segmentation_batch,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_segmentation_batch,
    )
    model_config = configuration["model"]
    model = build_model(
        model_config["architecture"],
        int(model_config["input_channels"]),
        int(model_config["num_classes"]),
        int(model_config.get("base_channels", 32)),
    )
    loss_config = configuration["loss"]
    criterion = CrossEntropyDiceLoss(
        float(loss_config.get("cross_entropy_weight", 1.0)),
        float(loss_config.get("dice_weight", 1.0)),
        bool(loss_config.get("include_background_in_dice", False)),
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return train_model(
        model,
        train_loader,
        validation_loader,
        criterion,
        configuration,
        Path(configuration["output"]["directory"]),
        device,
    )


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--smoke-steps", type=int, default=25)
    parser.add_argument("--fold", type=int, help="Override cross_validation.validation_fold")
    return parser.parse_args()


def main() -> None:
    """Run configured training or the synthetic overfit diagnostic."""
    args = parse_args()
    if args.smoke_test:
        initial, final = run_overfit_smoke_test(steps=args.smoke_steps)
        print(f"Overfit smoke test passed on CPU: loss {initial:.6f} -> {final:.6f}")
        return
    if args.config is None:
        raise SystemExit("--config is required unless --smoke-test is used")
    configuration = load_configuration(args.config)
    if args.fold is not None:
        configuration.setdefault("cross_validation", {})["enabled"] = True
        configuration["cross_validation"]["validation_fold"] = args.fold
    history = train_from_configuration(configuration)
    print(f"Completed {len(history)} training epochs")


if __name__ == "__main__":
    main()
