"""Synthetic annotation lifecycle, selection, and grouped-fold tests."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import tifffile

from astroseg.annotations import (
    import_annotation_pair,
    load_annotation_mask,
    save_pseudo_label_artifacts,
    select_uncertain_patches,
)
from astroseg.constants import MANIFEST_COLUMNS
from astroseg.io.manifest import validate_manifest
from astroseg.training.cross_validation import assign_grouped_folds, split_grouped_fold
from scripts.import_instance_annotations import discover_cellpose_pairs


def _manifest_row(image_id: str, status: str = "seed") -> dict[str, str]:
    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "image_id": image_id,
            "path": f"raw/{image_id}.tiff",
            "gfap_channel": "GFAP",
            "annotation_path": f"annotations/{image_id}.tiff" if status != "none" else "",
            "annotation_status": status,
            "split": "train",
        }
    )
    return row


def test_import_preserves_instance_mask_and_exports_binary_qc(tmp_path: Path) -> None:
    """Import archives the original values while producing an aligned binary target."""
    image_path = tmp_path / "image.ome.tiff"
    mask_path = tmp_path / "corrected_instances.tiff"
    gfap = np.arange(12 * 14, dtype=np.uint16).reshape(12, 14)
    tifffile.imwrite(
        image_path,
        np.stack((gfap, np.flipud(gfap))),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["GFAP", "DAPI"]}},
    )
    instance_mask = np.zeros((12, 14), dtype=np.uint16)
    instance_mask[2:5, 3:7] = 4
    instance_mask[7:10, 8:12] = 9
    tifffile.imwrite(mask_path, instance_mask)

    result = import_annotation_pair(
        "image",
        image_path,
        mask_path,
        "GFAP",
        tmp_path / "annotations",
        annotation_status="seed",
        annotator="tester",
    )
    np.testing.assert_array_equal(tifffile.imread(mask_path), instance_mask)
    np.testing.assert_array_equal(tifffile.imread(result.original_mask_path), instance_mask)
    assert set(np.unique(tifffile.imread(result.binary_mask_path))) == {0, 1}
    assert result.qc_overlay_path.is_file()
    assert result.annotation_status == "seed"


def test_cellpose_seg_npy_is_discovered_and_unwraps_masks(tmp_path: Path) -> None:
    """Cellpose dictionary exports should map by basename and expose instance IDs.

    The original ``*_seg.npy`` container remains untouched for provenance.
    """
    image_id = "BMP4_24h_20x_20240307_145"
    mask = np.zeros((9, 11), dtype=np.uint16)
    mask[2:6, 3:8] = 7
    export_path = tmp_path / f"{image_id}_seg.npy"
    np.save(export_path, {"masks": mask, "outlines": mask > 0}, allow_pickle=True)
    original_bytes = export_path.read_bytes()

    loaded = load_annotation_mask(export_path)
    pairs = discover_cellpose_pairs(pd.DataFrame({"image_id": [image_id]}), tmp_path)

    np.testing.assert_array_equal(loaded, mask)
    assert pairs.iloc[0]["image_id"] == image_id
    assert Path(pairs.iloc[0]["instance_mask_path"]) == export_path.resolve()
    assert export_path.read_bytes() == original_bytes


def test_import_rejects_dimension_mismatch_before_writing(tmp_path: Path) -> None:
    """A mask with a different pixel grid cannot be imported."""
    image_path = tmp_path / "image.ome.tiff"
    mask_path = tmp_path / "mask.npy"
    tifffile.imwrite(
        image_path,
        np.zeros((2, 10, 11), dtype=np.uint8),
        ome=True,
        metadata={"axes": "CYX", "Channel": {"Name": ["GFAP", "DAPI"]}},
    )
    np.save(mask_path, np.zeros((9, 11), dtype=np.uint8), allow_pickle=False)
    with pytest.raises(ValueError, match="does not match"):
        import_annotation_pair("image", image_path, mask_path, "GFAP", tmp_path / "annotations")
    assert not (tmp_path / "annotations").exists()


def test_manifest_rejects_unknown_annotation_state() -> None:
    """Annotation lifecycle values are explicit and validated."""
    row = _manifest_row("image")
    row["annotation_status"] = "maybe"
    with pytest.raises(ValueError, match="annotation_status"):
        validate_manifest(pd.DataFrame([row]))


def test_grouped_folds_keep_images_and_wells_together() -> None:
    """All images from a well occupy one fold and train/validation groups are disjoint."""
    rows = []
    for well_index in range(3):
        for image_index in range(2):
            row = _manifest_row(f"well{well_index}_image{image_index}")
            row["well_id"] = f"well{well_index}"
            rows.append(row)
    manifest = pd.DataFrame(rows)
    folded = assign_grouped_folds(manifest, n_splits=3, group_column="well_id", seed=9)
    assert (folded.groupby("well_id")["fold"].nunique() == 1).all()
    training, validation = split_grouped_fold(folded, 0, group_column="well_id")
    assert set(training["well_id"]).isdisjoint(validation["well_id"])
    assert set(training["image_id"]).isdisjoint(validation["image_id"])


def test_grouped_folds_balance_unequal_well_sizes() -> None:
    """Greedy fold assignment balances image counts without splitting wells.

    Wells containing five, four, and one image can be divided into two folds
    containing five images each while every well remains intact.
    """
    rows = []
    for well_id, image_count in (("large", 5), ("medium", 4), ("small", 1)):
        for image_index in range(image_count):
            row = _manifest_row(f"{well_id}_{image_index}")
            row["well_id"] = well_id
            rows.append(row)
    folded = assign_grouped_folds(
        pd.DataFrame(rows), n_splits=2, group_column="well_id", seed=3
    )
    assert sorted(folded["fold"].value_counts().tolist()) == [5, 5]
    assert (folded.groupby("well_id")["fold"].nunique() == 1).all()


def test_uncertainty_selection_prefers_ambiguous_predictions(tmp_path: Path) -> None:
    """A near-50/50 binary prediction ranks above a confident prediction."""
    probability_directory = tmp_path / "probabilities"
    probability_directory.mkdir()
    uncertain = np.full((2, 8, 8), 0.5, dtype=np.float32)
    confident = np.stack(
        (np.full((8, 8), 0.99, dtype=np.float32), np.full((8, 8), 0.01, dtype=np.float32))
    )
    np.save(probability_directory / "uncertain.npy", uncertain, allow_pickle=False)
    np.save(probability_directory / "confident.npy", confident, allow_pickle=False)
    manifest = pd.DataFrame(
        [_manifest_row("uncertain", "pseudo"), _manifest_row("confident", "pseudo")]
    )
    selected = select_uncertain_patches(
        manifest, probability_directory, patch_size=8, overlap=0, top_k=1
    )
    assert selected.iloc[0]["image_id"] == "uncertain"
    assert selected.iloc[0]["uncertainty"] == pytest.approx(1.0, abs=1e-5)


def test_pseudo_label_artifacts_are_separate_and_binary(tmp_path: Path) -> None:
    """Automatic probability and mask files are saved in dedicated subdirectories."""
    probabilities = np.stack(
        (np.full((6, 7), 0.25, dtype=np.float32), np.full((6, 7), 0.75, dtype=np.float32))
    )
    artifacts = save_pseudo_label_artifacts(
        "image", probabilities, np.ones((6, 7), dtype=np.uint16), tmp_path / "pseudo"
    )
    assert artifacts.probability_path.parent.name == "probabilities"
    assert artifacts.mask_path.parent.name == "masks"
    assert np.all(tifffile.imread(artifacts.mask_path) == 1)


def test_pseudo_label_artifacts_accept_explicit_heuristic_mask(tmp_path: Path) -> None:
    """A cleaned heuristic mask can override probability argmax for storage."""
    probabilities = np.stack(
        (np.full((5, 6), 0.75, dtype=np.float32), np.full((5, 6), 0.25, dtype=np.float32))
    )
    hard_mask = np.zeros((5, 6), dtype=np.uint8)
    hard_mask[1:4, 2:5] = 1

    artifacts = save_pseudo_label_artifacts(
        "heuristic",
        probabilities,
        np.ones((5, 6), dtype=np.uint16),
        tmp_path / "pseudo",
        hard_mask=hard_mask,
    )

    np.testing.assert_array_equal(tifffile.imread(artifacts.mask_path), hard_mask)
