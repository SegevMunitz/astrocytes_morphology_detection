"""Training loop for the nucleus-guided multi-head instance model."""

import csv
import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from astroseg.models import NucleusGuidedInstanceUNet
from astroseg.preprocessing import build_astrocyte_instance_targets
from astroseg.training.checkpoints import save_checkpoint
from astroseg.training.losses import NucleusGuidedInstanceLoss
from astroseg.training.metrics import metrics_from_logits


@dataclass(frozen=True)
class InstanceEpochMetrics:
    """Averages needed to diagnose and select an instance-training checkpoint."""

    total_loss: float
    semantic_dice: float
    boundary_dice: float
    consistency_loss: float
    semantic_loss: float
    boundary_loss: float
    offset_loss: float

    def __getitem__(self, index: int) -> float:
        """Retain index access for existing smoke-test reporting."""
        return (
            self.total_loss,
            self.semantic_dice,
            self.boundary_dice,
            self.consistency_loss,
            self.semantic_loss,
            self.boundary_loss,
            self.offset_loss,
        )[index]


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
    unlabeled_loader: DataLoader[Any] | None = None,
    teacher: nn.Module | None = None,
    consistency_weight: float = 0.0,
    consistency_confidence: float = 0.8,
    ema_decay: float = 0.99,
    gradient_scaler: torch.amp.GradScaler | None = None,
    mixed_precision: bool = False,
    auxiliary_dropout_probability: float = 0.0,
) -> InstanceEpochMetrics:
    """Run one train or validation epoch for all three prediction heads.

    Mean joint loss, foreground-compartment Dice, and boundary Dice are returned.
    Supplying an optimizer enables gradient updates; validation disables gradients.
    """
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_semantic_dice = 0.0
    total_boundary_dice = 0.0
    total_consistency = 0.0
    total_semantic_loss = 0.0
    total_boundary_loss = 0.0
    total_offset_loss = 0.0
    batches = 0
    unlabeled_iterator = iter(unlabeled_loader) if unlabeled_loader is not None else None
    for batch in tqdm(data_loader, desc="train" if training else "val", leave=False):
        images = batch["image"].to(device=device, dtype=torch.float32)
        targets = _move_targets(batch["targets"], device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training), torch.autocast(
            device_type=device.type,
            dtype=torch.float16,
            enabled=mixed_precision and device.type == "cuda",
        ):
            outputs = model(images)
            loss_components = criterion.components(outputs, targets)
            loss = loss_components["total"]
            consistency = torch.zeros((), device=device)
            if (
                training
                and unlabeled_loader is not None
                and teacher is not None
                and consistency_weight > 0
            ):
                assert unlabeled_iterator is not None
                try:
                    unlabeled_batch = next(unlabeled_iterator)
                except StopIteration:
                    unlabeled_iterator = iter(unlabeled_loader)
                    unlabeled_batch = next(unlabeled_iterator)
                unlabeled = unlabeled_batch["image"].to(
                    device=device, dtype=torch.float32
                )
                scale = torch.empty(
                    (unlabeled.shape[0], unlabeled.shape[1], 1, 1), device=device
                ).uniform_(0.8, 1.2)
                bias = torch.empty_like(scale).uniform_(-0.08, 0.08)
                noise = torch.randn_like(unlabeled) * 0.03
                strong = torch.clamp(unlabeled * scale + bias + noise, 0.0, 1.0)
                if auxiliary_dropout_probability > 0:
                    if strong.shape[1] != 3:
                        raise ValueError(
                            "Auxiliary-channel dropout requires exactly three inputs"
                        )
                    dropped = torch.rand(strong.shape[0], device=device) < auxiliary_dropout_probability
                    strong[dropped, 1] = 0
                with torch.no_grad():
                    teacher_outputs = teacher(unlabeled)
                student_outputs = model(strong)
                consistency_terms = []
                for head in ("semantic_logits", "boundary_logits"):
                    teacher_probability = torch.softmax(teacher_outputs[head], dim=1)
                    student_probability = torch.softmax(student_outputs[head], dim=1)
                    confident = (
                        teacher_probability.max(dim=1, keepdim=True).values
                        >= consistency_confidence
                    )
                    values = (student_probability - teacher_probability).square()
                    denominator = confident.sum().clamp_min(1) * values.shape[1]
                    consistency_terms.append((values * confident).sum() / denominator)
                consistency = torch.stack(consistency_terms).mean()
                loss = loss + consistency_weight * consistency
            if optimizer is not None:
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        "Training loss became NaN or infinite; aborting this run"
                    )
                if gradient_scaler is not None:
                    gradient_scaler.scale(loss).backward()
                    gradient_scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    gradient_scaler.step(optimizer)
                    gradient_scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                if teacher is not None:
                    with torch.no_grad():
                        for teacher_parameter, student_parameter in zip(
                            teacher.parameters(), model.parameters(), strict=True
                        ):
                            teacher_parameter.mul_(ema_decay).add_(
                                student_parameter, alpha=1.0 - ema_decay
                            )
        total_loss += float(loss.detach().cpu())
        total_semantic_loss += float(loss_components["semantic"].detach().cpu())
        total_boundary_loss += float(loss_components["boundary"].detach().cpu())
        total_offset_loss += float(loss_components["offset"].detach().cpu())
        total_consistency += float(consistency.detach().cpu())
        total_semantic_dice += metrics_from_logits(
            outputs["semantic_logits"].detach(), targets["semantic"]
        )["macro"]["dice"]
        total_boundary_dice += metrics_from_logits(
            outputs["boundary_logits"].detach(), targets["boundary"]
        )["macro"]["dice"]
        batches += 1
    if batches == 0:
        raise ValueError("Data loader produced no instance batches")
    return InstanceEpochMetrics(
        total_loss=total_loss / batches,
        semantic_dice=total_semantic_dice / batches,
        boundary_dice=total_boundary_dice / batches,
        consistency_loss=total_consistency / batches,
        semantic_loss=total_semantic_loss / batches,
        boundary_loss=total_boundary_loss / batches,
        offset_loss=total_offset_loss / batches,
    )


def train_instance_model(
    model: nn.Module,
    train_loader: DataLoader[Any],
    validation_loader: DataLoader[Any],
    criterion: NucleusGuidedInstanceLoss,
    configuration: dict[str, Any],
    output_directory: str | Path,
    device: torch.device,
    unlabeled_loader: DataLoader[Any] | None = None,
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
    scheduler = None
    if bool(training.get("lr_scheduler_enabled", True)):
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=float(training.get("lr_reduce_factor", 0.5)),
            patience=int(training.get("lr_patience", 5)),
            min_lr=float(training.get("minimum_learning_rate", 1e-6)),
        )
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    model.to(device)
    criterion.to(device)
    mixed_precision = bool(training.get("mixed_precision", True)) and device.type == "cuda"
    gradient_scaler = torch.amp.GradScaler("cuda", enabled=True) if mixed_precision else None
    semi_supervised = configuration.get("semi_supervised", {})
    augmentation = configuration.get("augmentation", {})
    teacher = None
    if unlabeled_loader is not None:
        teacher = copy.deepcopy(model).to(device)
        teacher.eval()
        for parameter in teacher.parameters():
            parameter.requires_grad_(False)
    history: list[dict[str, float | int]] = []
    best_metric = -1.0
    without_improvement = 0
    for epoch in range(1, int(training["epochs"]) + 1):
        ramp_epochs = max(1, int(semi_supervised.get("ramp_epochs", 10)))
        maximum_consistency = float(semi_supervised.get("consistency_weight", 0.25))
        consistency_weight = maximum_consistency * min(1.0, epoch / ramp_epochs)
        train_values = run_instance_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            unlabeled_loader=unlabeled_loader,
            teacher=teacher,
            consistency_weight=consistency_weight,
            consistency_confidence=float(semi_supervised.get("confidence_threshold", 0.8)),
            ema_decay=float(semi_supervised.get("ema_decay", 0.99)),
            gradient_scaler=gradient_scaler,
            mixed_precision=mixed_precision,
            auxiliary_dropout_probability=float(
                augmentation.get("auxiliary_dropout_probability", 0)
            ),
        )
        validation_values = run_instance_epoch(
            model,
            validation_loader,
            criterion,
            device,
            mixed_precision=mixed_precision,
        )
        row: dict[str, float | int] = {
            "epoch": epoch,
            "train_loss": train_values.total_loss,
            "train_semantic_loss": train_values.semantic_loss,
            "train_boundary_loss": train_values.boundary_loss,
            "train_offset_loss": train_values.offset_loss,
            "train_semantic_dice": train_values.semantic_dice,
            "train_boundary_dice": train_values.boundary_dice,
            "train_consistency_loss": train_values.consistency_loss,
            "consistency_weight": consistency_weight if unlabeled_loader is not None else 0.0,
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation_loss": validation_values.total_loss,
            "validation_semantic_loss": validation_values.semantic_loss,
            "validation_boundary_loss": validation_values.boundary_loss,
            "validation_offset_loss": validation_values.offset_loss,
            "validation_semantic_dice": validation_values.semantic_dice,
            "validation_boundary_dice": validation_values.boundary_dice,
        }
        history.append(row)
        checkpoint_metric = str(
            training.get("checkpoint_metric", "validation_joint_dice")
        )
        available_metrics = {
            "validation_semantic_dice": validation_values.semantic_dice,
            "validation_boundary_dice": validation_values.boundary_dice,
            "validation_joint_dice": (
                validation_values.semantic_dice + validation_values.boundary_dice
            )
            / 2.0,
        }
        if checkpoint_metric not in available_metrics:
            raise ValueError(
                f"Unknown training.checkpoint_metric {checkpoint_metric!r}; "
                f"expected one of {sorted(available_metrics)}"
            )
        metric = available_metrics[checkpoint_metric]
        row["checkpoint_metric"] = metric
        if scheduler is not None:
            scheduler.step(metric)
        save_checkpoint(destination / "last.pt", model, optimizer, epoch, metric, configuration)
        save_every = int(training.get("save_every", 10))
        if save_every <= 0:
            raise ValueError("training.save_every must be positive")
        if epoch % save_every == 0 or epoch == int(training["epochs"]):
            save_checkpoint(
                destination / f"epoch_{epoch:04d}.pt",
                model,
                optimizer,
                epoch,
                metric,
                configuration,
            )
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
