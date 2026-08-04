"""Training loop for the nucleus-guided multi-head instance model."""

import csv
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm

from astroseg.models import NucleusGuidedInstanceUNet
from astroseg.preprocessing import build_astrocyte_instance_targets
from astroseg.training.checkpoints import save_checkpoint
from astroseg.training.losses import NucleusGuidedInstanceLoss
from astroseg.training.metrics import metrics_from_logits


def _move_targets(
    targets: dict[str, torch.Tensor], device: torch.device
) -> dict[str, torch.Tensor]:
    """Move each multi-head target tensor to one training device.

    Integer semantic/boundary labels retain their dtype. Offset planes and masks
    are converted to float32 to match the regression head and avoid implicit casts.
    """
    return {
        "semantic": targets["semantic"].to(device=device, dtype=torch.long),
        "boundary": targets["boundary"].to(device=device, dtype=torch.long),
        "offsets": targets["offsets"].to(device=device, dtype=torch.float32),
        "offset_mask": targets["offset_mask"].to(device=device, dtype=torch.float32),
    }


def run_instance_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: NucleusGuidedInstanceLoss,
    device: torch.device,
    optimizer: AdamW | None = None,
) -> tuple[float, float, float]:
    """Run one train or validation epoch for all three prediction heads.

    Mean joint loss, foreground-compartment Dice, and boundary Dice are returned.
    Supplying an optimizer enables gradient updates; validation disables gradients.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_semantic_dice = 0.0
    total_boundary_dice = 0.0
    batches = 0
    for batch in tqdm(data_loader, desc="train" if training else "val", leave=False):
        images = batch["image"].to(device=device, dtype=torch.float32)
        targets = _move_targets(batch["targets"], device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            outputs = model(images)
            loss = criterion(outputs, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_semantic_dice += metrics_from_logits(
            outputs["semantic_logits"].detach(), targets["semantic"]
        )["macro"]["dice"]
        total_boundary_dice += metrics_from_logits(
            outputs["boundary_logits"].detach(), targets["boundary"]
        )["macro"]["dice"]
        batches += 1
    if batches == 0:
        raise ValueError("Data loader produced no instance batches")
    return (
        total_loss / batches,
        total_semantic_dice / batches,
        total_boundary_dice / batches,
    )


def train_instance_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: NucleusGuidedInstanceLoss,
    configuration: dict[str, Any],
    output_directory: str | Path,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Train with early stopping on validation compartment Dice.

    Best/latest checkpoints and a CSV history preserve the same operational
    contract as the semantic trainer while recording both semantic and boundary QC.
    """
    training = configuration["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.to(device)
    history: list[dict[str, float | int]] = []
    best_metric = -1.0
    without_improvement = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        train_values = run_instance_epoch(model, train_loader, criterion, device, optimizer)
        validation_values = run_instance_epoch(model, validation_loader, criterion, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_values[0],
            "train_semantic_dice": train_values[1],
            "train_boundary_dice": train_values[2],
            "validation_loss": validation_values[0],
            "validation_semantic_dice": validation_values[1],
            "validation_boundary_dice": validation_values[2],
        }
        history.append(row)
        metric = validation_values[1]
        save_checkpoint(destination / "last.pt", model, optimizer, epoch, metric, configuration)
        if metric > best_metric:
            best_metric = metric
            without_improvement = 0
            save_checkpoint(destination / "best.pt", model, optimizer, epoch, metric, configuration)
        else:
            without_improvement += 1
        with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if without_improvement >= int(training["early_stopping_patience"]):
            break
    return history


def run_instance_overfit_smoke_test(steps: int = 20) -> tuple[float, float]:
    """Overfit one synthetic two-cell sample to verify all heads and gradients.

    This is an engineering diagnostic only. It does not test biological process
    ownership or replace validation on human-corrected complete-cell instances.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    size = 32
    yy, xx = np.indices((size, size))
    nuclei = np.zeros((size, size), dtype=np.uint16)
    nuclei[(yy - 10) ** 2 + (xx - 8) ** 2 <= 2**2] = 11
    nuclei[(yy - 21) ** 2 + (xx - 23) ** 2 <= 2**2] = 22
    cells = np.zeros((size, size), dtype=np.uint16)
    cells[((yy - 10) ** 2 + (xx - 8) ** 2 <= 6**2) | ((yy == 10) & (xx < 20))] = 1
    cells[((yy - 21) ** 2 + (xx - 23) ** 2 <= 6**2) | ((xx == 23) & (yy > 8))] = 2
    targets = build_astrocyte_instance_targets(cells, nuclei, soma_radius=4, offset_scale=32)
    nucleus_binary = (nuclei > 0).astype(np.float32)
    gfap = (cells > 0).astype(np.float32)
    distance = np.clip(1.0 - np.sqrt((yy - 16) ** 2 + (xx - 16) ** 2) / 24, 0, 1)
    image = torch.from_numpy(np.stack((gfap, nucleus_binary, distance)).astype(np.float32))[None]
    target_tensors = {
        "semantic": torch.from_numpy(targets.semantic)[None],
        "boundary": torch.from_numpy(targets.boundary)[None],
        "offsets": torch.from_numpy(targets.offsets)[None],
        "offset_mask": torch.from_numpy(targets.offset_mask)[None],
    }
    model = NucleusGuidedInstanceUNet(input_channels=3, base_channels=2)
    criterion = NucleusGuidedInstanceLoss()
    optimizer = AdamW(model.parameters(), lr=0.02, weight_decay=0)
    initial = float("nan")
    final = float("nan")
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(image), target_tensors)
        if step == 0:
            initial = float(loss.detach())
        loss.backward()
        optimizer.step()
        final = float(loss.detach())
    if not final < initial:
        raise RuntimeError(f"Instance smoke test did not reduce loss: {initial:.6f} -> {final:.6f}")
    return initial, final
