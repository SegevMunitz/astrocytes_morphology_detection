"""Run one supervised plus consistency update on real multichannel data."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

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
from astroseg.training import NucleusGuidedInstanceLoss, run_instance_epoch


def run_real_smoke_test(configuration: dict) -> tuple[float, float, float, float]:
    """Load one labeled/unlabeled image and execute one complete optimizer step."""
    data = configuration["data"]
    manifest_path = Path(data["manifest_path"])
    manifest = load_manifest(manifest_path)
    labeled = manifest.loc[
        manifest["annotation_status"].isin(TRAINABLE_ANNOTATION_STATUSES)
        & (manifest["instance_annotation_path"].astype(str).str.strip() != "")
    ].iloc[[0]].copy()
    unlabeled = manifest.loc[
        (manifest["split"] == "train") & (manifest["annotation_status"] == "none")
    ].iloc[[0]].copy()
    labeled["split"] = "train"
    common = {
        "patch_size": int(data["patch_size"]),
        "overlap": int(data["overlap"]),
        "max_nucleus_distance": float(data.get("max_nucleus_distance", 64)),
        "manifest_base_directory": manifest_path.parent,
        "input_mode": str(data.get("input_mode", "fluorescence")),
    }
    supervised_dataset = AstrocyteInstanceDataset(
        labeled,
        "train",
        soma_radius=float(data.get("soma_radius", 20)),
        offset_scale=float(data.get("offset_scale", 256)),
        augmentation=RandomInstanceAugmentation(
            auxiliary_dropout_probability=float(
                configuration.get("augmentation", {}).get(
                    "auxiliary_dropout_probability", 0
                )
            )
        ),
        annotation_statuses=TRAINABLE_ANNOTATION_STATUSES,
        **common,
    )
    unlabeled_dataset = AstrocyteUnlabeledDataset(unlabeled, **common)
    supervised_loader = DataLoader(
        Subset(supervised_dataset, [0]),
        batch_size=1,
        collate_fn=collate_instance_batch,
    )
    unlabeled_loader = DataLoader(
        Subset(unlabeled_dataset, [0]),
        batch_size=1,
        collate_fn=collate_unlabeled_batch,
    )
    model_configuration = configuration["model"]
    model = build_model(
        str(model_configuration["architecture"]),
        int(model_configuration["input_channels"]),
        int(model_configuration["num_classes"]),
        int(model_configuration.get("base_channels", 32)),
    )
    teacher = copy.deepcopy(model)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    loss_configuration = configuration["loss"]
    criterion = NucleusGuidedInstanceLoss(
        float(loss_configuration.get("semantic_weight", 1)),
        float(loss_configuration.get("boundary_weight", 1)),
        float(loss_configuration.get("offset_weight", 1)),
        loss_configuration.get("semantic_class_weights"),
        loss_configuration.get("boundary_class_weights"),
    )
    optimizer = AdamW(model.parameters(), lr=1e-4)
    return run_instance_epoch(
        model,
        supervised_loader,
        criterion,
        torch.device("cpu"),
        optimizer,
        unlabeled_loader=unlabeled_loader,
        teacher=teacher,
        consistency_weight=0.025,
    )


def main() -> None:
    """Parse the cluster configuration and report the four smoke metrics."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    values = run_real_smoke_test(load_yaml_configuration(args.config))
    print(
        "Real multichannel smoke passed: "
        f"loss={values[0]:.6f}, semantic_dice={values[1]:.4f}, "
        f"boundary_dice={values[2]:.4f}, consistency={values[3]:.6f}"
    )


if __name__ == "__main__":
    main()
