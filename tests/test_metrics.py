"""Manually verifiable model and metric tests."""

import numpy as np
import pytest
import torch

from astroseg.models import UNet, build_model
from astroseg.training.losses import CrossEntropyDiceLoss, DiceLoss
from astroseg.training.metrics import metrics_from_logits, metrics_from_predictions


def test_unet_preserves_odd_spatial_shape() -> None:
    """Interpolation around skip connections preserves odd input dimensions."""
    model = UNet(input_channels=3, num_classes=2, base_channels=4)
    output = model(torch.randn(2, 3, 33, 47))
    assert output.shape == (2, 2, 33, 47)
    assert not any(isinstance(layer, torch.nn.Linear) for layer in model.modules())


def test_model_factory_segformer_placeholder() -> None:
    """The factory supports U-Net and fails clearly for deferred SegFormer work."""
    assert isinstance(build_model("unet", 3, 2, 4), UNet)
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_model("segformer", 3, 2)


def test_metrics_match_manual_foreground_values() -> None:
    """Dice, IoU, precision, and recall match a two-by-two example."""
    prediction = np.array([[0, 1], [0, 1]])
    target = np.array([[0, 1], [1, 1]])
    metrics = metrics_from_predictions(prediction, target, num_classes=2)
    foreground = metrics["per_class"][1]
    assert foreground["dice"] == pytest.approx(0.8)
    assert foreground["iou"] == pytest.approx(2 / 3)
    assert foreground["precision"] == pytest.approx(1.0)
    assert foreground["recall"] == pytest.approx(2 / 3)
    assert metrics["macro"] == pytest.approx(foreground)


def test_empty_classes_are_nan_safe_and_logits_api_matches() -> None:
    """Absent classes produce defined values and the logits API uses argmax masks."""
    target = torch.zeros((1, 3, 3), dtype=torch.long)
    logits = torch.zeros((1, 2, 3, 3))
    logits[:, 0] = 1
    metrics = metrics_from_logits(logits, target)
    assert metrics["per_class"][1]["dice"] == 1.0
    assert np.isfinite(list(metrics["macro"].values())).all()


def test_multiclass_losses_are_scalar_and_finite() -> None:
    """Dice and combined losses support more than two classes."""
    logits = torch.randn(2, 3, 8, 9, requires_grad=True)
    target = torch.randint(0, 3, (2, 8, 9))
    for criterion in (DiceLoss(), CrossEntropyDiceLoss()):
        loss = criterion(logits, target)
        assert loss.ndim == 0 and torch.isfinite(loss)
