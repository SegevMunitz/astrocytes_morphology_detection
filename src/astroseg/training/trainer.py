"""Minimal deterministic PyTorch training loop."""

import csv
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from astroseg.models.unet import UNet
from astroseg.training.checkpoints import save_checkpoint
from astroseg.training.losses import CrossEntropyDiceLoss
from astroseg.training.metrics import metrics_from_logits


def set_deterministic_seed(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible baseline experiments.

    CUDA generators are seeded when available, and deterministic algorithms are
    requested in warning mode so unsupported operations remain visible.
    """
    if seed < 0:
        raise ValueError("seed must be non-negative")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
    optimizer: AdamW | None,
    description: str,
) -> tuple[float, float]:
    """Execute the shared batch loop for either training or validation.

    Supplying an optimizer enables gradients and parameter updates; ``None`` uses
    evaluation mode. Mean batch loss and foreground Dice are returned.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_dice = 0.0
    batches = 0
    for batch in tqdm(data_loader, desc=description, leave=False):
        if isinstance(batch, dict):
            images, targets = batch["image"], batch["target"]
        else:
            images, targets = batch[0], batch[1]
        images = images.to(device=device, dtype=torch.float32)
        targets = targets.to(device=device, dtype=torch.long)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, targets)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.detach().cpu())
        total_dice += metrics_from_logits(logits.detach(), targets)["macro"]["dice"]
        batches += 1
    if batches == 0:
        raise ValueError("Data loader produced no batches")
    return total_loss / batches, total_dice / batches


def train_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    optimizer: AdamW,
    device: torch.device,
) -> tuple[float, float]:
    """Run one training epoch with gradient updates over every loader batch.

    The function moves tensors to the selected device, backpropagates the loss,
    updates AdamW, and returns mean batch loss and foreground macro Dice.
    """
    return _run_epoch(model, data_loader, criterion, device, optimizer, "train")


def validation_epoch(
    model: nn.Module,
    data_loader: DataLoader[Any],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Evaluate one loader epoch without gradients or optimizer updates.

    The model enters evaluation mode and returns mean batch loss together with
    foreground macro Dice for checkpoint selection and early stopping.
    """
    return _run_epoch(model, data_loader, criterion, device, None, "val")


def train_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: nn.Module,
    configuration: dict[str, Any],
    output_directory: str | Path,
    device: torch.device,
) -> list[dict[str, float | int]]:
    """Train a segmentation model with AdamW and validation-based early stopping.

    Best and latest checkpoints include complete configuration and optimizer state,
    while a CSV history records loss and Dice values for every completed epoch.
    """
    training_config = configuration["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config["weight_decay"]),
    )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.to(device)
    best_metric = -1.0
    epochs_without_improvement = 0
    history: list[dict[str, float | int]] = []
    epochs = int(training_config["epochs"])
    patience = int(training_config["early_stopping_patience"])
    for epoch in range(1, epochs + 1):
        train_loss, train_dice = train_epoch(model, train_loader, criterion, optimizer, device)
        validation_loss, validation_dice = validation_epoch(model, validation_loader, criterion, device)
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_dice": train_dice,
            "validation_loss": validation_loss,
            "validation_dice": validation_dice,
        }
        history.append(row)
        save_checkpoint(destination / "last.pt", model, optimizer, epoch, validation_dice, configuration)
        if validation_dice > best_metric:
            best_metric = validation_dice
            epochs_without_improvement = 0
            save_checkpoint(destination / "best.pt", model, optimizer, epoch, validation_dice, configuration)
        else:
            epochs_without_improvement += 1
        with (destination / "history.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        if epochs_without_improvement >= patience:
            break
    return history


def run_overfit_smoke_test(seed: int = 42, steps: int = 25) -> tuple[float, float]:
    """Run a CPU diagnostic that overfits one deterministic synthetic patch.

    This checks the model, loss, backward pass, and optimizer plumbing without
    claiming scientific performance. Failure to reduce loss raises an error.
    """
    if steps < 2:
        raise ValueError("steps must be at least 2")
    set_deterministic_seed(seed)
    yy, xx = torch.meshgrid(torch.arange(32), torch.arange(32), indexing="ij")
    target = (((yy - 16) ** 2 + (xx - 16) ** 2) < 8**2).long().unsqueeze(0)
    nucleus = (((yy - 16) ** 2 + (xx - 16) ** 2) < 3**2).float()
    gfap = target.squeeze(0).float() * 0.9 + 0.05
    proximity = torch.clamp(1.0 - torch.sqrt((yy - 16).float() ** 2 + (xx - 16).float() ** 2) / 16, 0, 1)
    inputs = torch.stack((gfap, nucleus, proximity)).unsqueeze(0)
    loader = DataLoader(TensorDataset(inputs, target), batch_size=1, shuffle=False)
    model = UNet(input_channels=3, num_classes=2, base_channels=4)
    criterion = CrossEntropyDiceLoss()
    optimizer = AdamW(model.parameters(), lr=0.01, weight_decay=0.0)
    model.train()
    initial_loss: float | None = None
    final_loss = float("inf")
    for _ in range(steps):
        batch_inputs, batch_target = next(iter(loader))
        optimizer.zero_grad(set_to_none=True)
        logits = model(batch_inputs)
        loss = criterion(logits, batch_target)
        if initial_loss is None:
            initial_loss = float(loss.detach())
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach())
    assert initial_loss is not None
    if final_loss >= initial_loss:
        raise RuntimeError(f"Overfit smoke test did not reduce loss: {initial_loss:.6f} -> {final_loss:.6f}")
    return initial_loss, final_loss
