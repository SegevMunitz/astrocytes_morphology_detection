"""Synthetic regression tests for complete-cell instance segmentation."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile
import torch

from astroseg.annotations import import_astrocyte_instance_pair
from astroseg.constants import MANIFEST_COLUMNS
from astroseg.datasets import (
    AstrocyteInstanceDataset,
    RandomInstanceAugmentation,
    RandomInstanceFlip,
)
from astroseg.models import MultichannelInstanceUNet, NucleusGuidedInstanceUNet
from astroseg.postprocessing import separate_astrocyte_instances
from astroseg.preprocessing import build_astrocyte_instance_targets
from astroseg.training import (
    NucleusGuidedInstanceLoss,
    instance_segmentation_metrics,
    process_ownership_accuracy,
)
from scripts.train_instances import _eligible_instance_rows


def _two_cell_labels() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create two complete astrocytes with known nuclei and compartments.

    The tiny arrays keep target and metric tests deterministic and inexpensive.
    """
    yy, xx = np.indices((24, 24))
    nuclei = np.zeros((24, 24), dtype=np.uint16)
    nuclei[(yy - 7) ** 2 + (xx - 6) ** 2 <= 2**2] = 10
    nuclei[(yy - 16) ** 2 + (xx - 17) ** 2 <= 2**2] = 20
    cells = np.zeros_like(nuclei)
    cells[((yy - 7) ** 2 + (xx - 6) ** 2 <= 5**2) | ((yy == 7) & (xx < 14))] = 1
    cells[((yy - 16) ** 2 + (xx - 17) ** 2 <= 5**2) | ((xx == 17) & (yy > 8))] = 2
    compartments = np.zeros_like(nuclei, dtype=np.uint8)
    compartments[cells > 0] = 3
    compartments[((cells == 1) & ((yy - 7) ** 2 + (xx - 6) ** 2 <= 4**2))] = 2
    compartments[((cells == 2) & ((yy - 16) ** 2 + (xx - 17) ** 2 <= 4**2))] = 2
    compartments[nuclei > 0] = 1
    return cells, nuclei, compartments


def test_instance_targets_encode_compartments_boundaries_and_nucleus_offsets() -> None:
    """Every cell should map to one nucleus and point its pixels toward it.

    Explicit compartment labels must pass through unchanged while offset targets
    remain normalized on the complete cell support.
    """
    cells, nuclei, compartments = _two_cell_labels()
    targets = build_astrocyte_instance_targets(
        cells, nuclei, compartments, offset_scale=24
    )
    assert targets.cell_to_nucleus == {1: 10, 2: 20}
    np.testing.assert_array_equal(targets.semantic, compartments)
    assert targets.boundary.shape == cells.shape
    assert targets.boundary[7, 12] == 0  # process/background edge is not a cell boundary
    assert targets.offsets.shape == (2, *cells.shape)
    assert np.all(targets.offset_mask[cells > 0] == 1)
    assert targets.offsets[1, 7, 12] < 0
    long_targets = build_astrocyte_instance_targets(
        cells, nuclei, compartments, offset_scale=2
    )
    assert np.abs(long_targets.offsets).max() > 1


def test_instance_dataset_loads_aligned_complete_cell_targets(tmp_path: Path) -> None:
    """A manifest instance row should produce all multi-head patch tensors.

    Original cell IDs remain available separately from derived model targets.
    """
    cells, nuclei, compartments = _two_cell_labels()
    image_path = tmp_path / "image.ome.tiff"
    nucleus_path = tmp_path / "nuclei.tiff"
    instance_path = tmp_path / "cells.tiff"
    compartment_path = tmp_path / "compartments.tiff"
    binary_path = tmp_path / "binary.tiff"
    gfap = (cells > 0).astype(np.uint16) * 100
    tifffile.imwrite(
        image_path,
        np.stack((gfap, (nuclei > 0).astype(np.uint16) * 100)),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["GFAP", "DAPI"]}},
    )
    for path, value in (
        (nucleus_path, nuclei),
        (instance_path, cells),
        (compartment_path, compartments),
        (binary_path, (cells > 0).astype(np.uint8)),
    ):
        tifffile.imwrite(path, value)
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "image_id": "synthetic",
            "path": str(image_path),
            "gfap_channel": "GFAP",
            "dapi_channel": "DAPI",
            "cellpose_mask_path": str(nucleus_path),
            "annotation_path": str(binary_path),
            "instance_annotation_path": str(instance_path),
            "compartment_annotation_path": str(compartment_path),
            "annotation_status": "seed",
            "split": "train",
        }
    )
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)
    dataset = AstrocyteInstanceDataset(manifest_path, "train", patch_size=16, overlap=4)
    item = dataset[0]
    assert item["image"].shape == (3, 16, 16)
    assert item["targets"]["semantic"].shape == (16, 16)
    assert item["targets"]["offsets"].shape == (2, 16, 16)
    assert item["instances"].dtype == torch.int64


def test_cellpose_instance_import_writes_plain_training_tiff(tmp_path: Path) -> None:
    """Pickled Cellpose sources must not become dataset training paths.

    The original dictionary remains archived, while the normalized instance path
    is a non-pickled 2D TIFF that the training dataset can load safely.
    """
    cells, nuclei, _ = _two_cell_labels()
    image_path = tmp_path / "image.ome.tiff"
    nucleus_path = tmp_path / "nuclei.tiff"
    cellpose_path = tmp_path / "image_seg.npy"
    tifffile.imwrite(
        image_path,
        np.stack(((cells > 0).astype(np.uint8), (nuclei > 0).astype(np.uint8))),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["GFAP", "DAPI"]}},
    )
    tifffile.imwrite(nucleus_path, nuclei)
    np.save(cellpose_path, {"masks": cells, "outlines": cells > 0}, allow_pickle=True)

    result = import_astrocyte_instance_pair(
        "image",
        image_path,
        nucleus_path,
        cellpose_path,
        "GFAP",
        tmp_path / "annotations",
    )

    assert result.base.original_mask_path.suffix == ".npy"
    assert result.instance_mask_path.suffix == ".tiff"
    np.testing.assert_array_equal(tifffile.imread(result.instance_mask_path), cells)


def test_instance_training_never_selects_reserved_test_images(tmp_path: Path) -> None:
    """Human-labeled test rows must remain outside grouped model development.

    Annotation availability cannot override an explicit held-out test assignment.
    """
    rows = []
    for image_id, split in (("development", "train"), ("held_out", "test")):
        row = {column: "" for column in MANIFEST_COLUMNS}
        row.update(
            {
                "image_id": image_id,
                "path": f"data/raw/{image_id}.tif",
                "annotation_path": f"data/annotations/{image_id}_binary.tiff",
                "instance_annotation_path": f"data/annotations/{image_id}_cells.tiff",
                "annotation_status": "reviewed",
                "split": split,
            }
        )
        rows.append(row)
    manifest_path = tmp_path / "manifest.csv"
    pd.DataFrame(rows, columns=MANIFEST_COLUMNS).to_csv(manifest_path, index=False)

    selected = _eligible_instance_rows(manifest_path, {"reviewed"})

    assert selected["image_id"].tolist() == ["development"]


def test_boundary_target_marks_only_contacts_between_different_cells() -> None:
    """Touching cell IDs are separated without labeling their outer GFAP edge.

    This preserves one-pixel processes while still supervising true cell contacts.
    """
    cells = np.zeros((8, 8), dtype=np.uint8)
    cells[2:6, 1:4] = 1
    cells[2:6, 4:7] = 2
    nuclei = np.zeros_like(cells)
    nuclei[3, 2] = 10
    nuclei[3, 5] = 20
    targets = build_astrocyte_instance_targets(cells, nuclei, soma_radius=2, offset_scale=8)
    assert np.all(targets.boundary[2:6, 3:5] == 1)
    assert targets.boundary[2, 1] == 0
    assert targets.boundary[5, 6] == 0


def test_instance_targets_assign_nearby_nonoverlapping_nuclei() -> None:
    """A GFAP-negative nuclear hole must still anchor its astrocyte.

    One distant cell remains useful for semantic and boundary supervision but is
    excluded from ownership-offset loss because its nucleus identity is unknown.
    """
    cells = np.zeros((32, 32), dtype=np.uint16)
    cells[5:13, 5:13] = 1
    cells[24:29, 24:29] = 2
    cells[8:10, 8:10] = 0
    nuclei = np.zeros_like(cells)
    nuclei[8:10, 8:10] = 7

    targets = build_astrocyte_instance_targets(
        cells, nuclei, soma_radius=3, max_nucleus_distance=8
    )

    assert targets.cell_to_nucleus == {1: 7}
    assert np.all(targets.semantic[nuclei == 7] == 1)
    assert np.all(targets.instances[nuclei == 7] == 1)
    assert np.all(targets.offset_mask[cells == 2] == 0)
    assert np.all(targets.semantic[cells == 2] == 3)


def test_instance_model_and_loss_cover_all_three_heads() -> None:
    """The multi-head U-Net should preserve odd spatial sizes and backpropagate.

    This catches incompatible model, target, or loss API changes early.
    """
    model = NucleusGuidedInstanceUNet(input_channels=3, base_channels=2)
    inputs = torch.rand(1, 3, 25, 27)
    outputs = model(inputs)
    assert outputs["semantic_logits"].shape == (1, 4, 25, 27)
    assert outputs["boundary_logits"].shape == (1, 2, 25, 27)
    assert outputs["offsets"].shape == (1, 2, 25, 27)
    targets = {
        "semantic": torch.randint(0, 4, (1, 25, 27)),
        "boundary": torch.randint(0, 2, (1, 25, 27)),
        "offsets": torch.zeros(1, 2, 25, 27),
        "offset_mask": torch.ones(1, 25, 27),
    }
    loss = NucleusGuidedInstanceLoss()(outputs, targets)
    loss.backward()
    assert torch.isfinite(loss)
    assert model.offset_output.weight.grad is not None


def test_multichannel_residual_model_requires_three_planes_and_all_heads() -> None:
    """The production model preserves odd image sizes and its input contract."""
    model = MultichannelInstanceUNet(base_channels=8)
    outputs = model(torch.rand(1, 3, 33, 35))
    assert outputs["semantic_logits"].shape == (1, 4, 33, 35)
    assert outputs["boundary_logits"].shape == (1, 2, 33, 35)
    assert outputs["offsets"].shape == (1, 2, 33, 35)
    with pytest.raises(ValueError, match="exactly three"):
        MultichannelInstanceUNet(input_channels=4)


def test_instance_flip_corrects_ownership_vector_signs() -> None:
    """Spatial flips must negate only the matching offset component.

    Masks and instance IDs are transformed in exact alignment with the input.
    """
    image = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    offsets = np.zeros((2, 3, 4), dtype=np.float32)
    offsets[0] = 2
    offsets[1] = 3
    targets = {
        "semantic": np.arange(12).reshape(3, 4),
        "boundary": np.zeros((3, 4), dtype=np.uint8),
        "offsets": offsets,
        "offset_mask": np.ones((3, 4), dtype=np.float32),
        "instances": np.ones((3, 4), dtype=np.uint16),
    }
    transformed_image, transformed = RandomInstanceFlip(1, 1)(image, targets)
    np.testing.assert_array_equal(transformed_image, image[..., ::-1, ::-1])
    assert np.all(transformed["offsets"][0] == -2)
    assert np.all(transformed["offsets"][1] == -3)


def test_instance_augmentation_can_drop_only_the_auxiliary_plane() -> None:
    """GFP dropout should reproduce two-channel acquisitions without misalignment."""
    image = np.stack(
        [np.full((3, 4), value, dtype=np.float32) for value in (0.2, 0.4, 0.6)]
    )
    targets = {
        "semantic": np.zeros((3, 4), dtype=np.uint8),
        "boundary": np.zeros((3, 4), dtype=np.uint8),
        "offsets": np.zeros((2, 3, 4), dtype=np.float32),
        "offset_mask": np.zeros((3, 4), dtype=np.float32),
        "instances": np.zeros((3, 4), dtype=np.uint16),
    }
    augmentation = RandomInstanceAugmentation(
        horizontal_probability=0,
        vertical_probability=0,
        rotation_probability=0,
        intensity_probability=0,
        auxiliary_dropout_probability=1,
    )
    transformed, transformed_targets = augmentation(image, targets)
    np.testing.assert_array_equal(transformed[0], image[0])
    np.testing.assert_array_equal(transformed[1], 0)
    np.testing.assert_array_equal(transformed[2], image[2])
    np.testing.assert_array_equal(transformed_targets["semantic"], targets["semantic"])


def test_learned_offsets_assign_long_processes_to_their_nuclei() -> None:
    """Ownership vectors should override geometric proximity along long processes.

    The left cell reaches near the right nucleus and vice versa, representing the
    failure mode that nearest-nucleus assignment alone cannot solve.
    """
    shape = (24, 32)
    nuclei = np.zeros(shape, dtype=np.uint16)
    nuclei[10, 5] = 11
    nuclei[10, 26] = 22
    foreground = np.zeros(shape, dtype=np.float32)
    foreground[8:10, 4:24] = 1
    foreground[12:14, 8:28] = 1
    offsets = np.zeros((2, *shape), dtype=np.float32)
    yy, xx = np.indices(shape)
    left_support = foreground > 0
    left_support[11:] = False
    right_support = foreground > 0
    right_support[:11] = False
    offsets[0, left_support] = (10 - yy[left_support]) / 32
    offsets[1, left_support] = (5 - xx[left_support]) / 32
    offsets[0, right_support] = (10 - yy[right_support]) / 32
    offsets[1, right_support] = (26 - xx[right_support]) / 32
    result = separate_astrocyte_instances(
        foreground,
        nuclei,
        ownership_offsets=offsets,
        offset_scale=32,
        max_offset_endpoint_distance=4,
        max_nucleus_to_gfap_distance=4,
        soma_expansion=1,
        min_cell_area=1,
        min_gfap_area=1,
    )
    left_cell = result.nucleus_cell_labels[10, 5]
    right_cell = result.nucleus_cell_labels[10, 26]
    assert result.labels[8, 22] == left_cell
    assert result.labels[13, 9] == right_cell
    assert result.ownership_mode == "learned_offsets_with_watershed_fallback"


def test_instance_metrics_penalize_wrong_process_owner() -> None:
    """Object overlap and process ownership are evaluated as distinct outcomes.

    Swapping one process region lowers ownership accuracy even when cell objects
    remain present and matchable.
    """
    target, _, compartments = _two_cell_labels()
    predicted = target.copy()
    process = (compartments == 3) & (target == 1)
    predicted[process] = 2
    metrics = instance_segmentation_metrics(predicted, target, iou_threshold=0.2)
    assert metrics["true_positive"] == 2
    assert process_ownership_accuracy(predicted, target, compartments) < 1
