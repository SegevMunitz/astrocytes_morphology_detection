"""Checkpoint serialization helpers."""

from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: Optimizer,
    epoch: int,
    validation_metric: float,
    configuration: dict[str, Any],
) -> None:
    """Persist all state required to inspect or resume a training run.

    The checkpoint contains model and optimizer states, completed epoch, tracked
    validation metric, and the complete configuration used to build the run.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "validation_metric": validation_metric,
            "configuration": configuration,
        },
        destination,
    )


def load_checkpoint(path: str | Path, device: torch.device) -> dict[str, Any]:
    """Load a checkpoint onto the requested device and validate its schema.

    Missing files or required state fields produce explicit errors before model
    restoration. The raw checkpoint mapping is returned to the caller.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {source}")
    checkpoint = torch.load(source, map_location=device, weights_only=False)
    required = {"model_state", "optimizer_state", "epoch", "validation_metric", "configuration"}
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(f"Checkpoint is missing required fields: {sorted(missing)}")
    return checkpoint
