"""Tests for portable configuration and cluster-storage manifest relocation."""

from pathlib import Path

import pandas as pd

from astroseg.constants import MANIFEST_COLUMNS
from astroseg.io import load_yaml_configuration
from scripts.relocate_runtime_manifest import relocate_runtime_manifest


def test_yaml_configuration_expands_cluster_data_root(
    tmp_path: Path, monkeypatch,
) -> None:
    """Cluster paths should expand without embedding one HUJI account location.

    Nested lists are included to protect the recursive expansion used by future
    multi-dataset configuration values.
    """
    monkeypatch.setenv("ASTROSEG_DATA_ROOT", tmp_path.as_posix())
    source = tmp_path / "config.yaml"
    source.write_text(
        "data:\n  manifest_path: ${ASTROSEG_DATA_ROOT}/outputs/manifest.csv\n"
        "output:\n  directories:\n    - ${ASTROSEG_DATA_ROOT}/outputs\n",
        encoding="utf-8",
    )

    configuration = load_yaml_configuration(source, required_sections=("data", "output"))

    assert configuration["data"]["manifest_path"] == (
        tmp_path / "outputs" / "manifest.csv"
    ).as_posix()
    assert configuration["output"]["directories"] == [(tmp_path / "outputs").as_posix()]


def test_relocate_runtime_manifest_maps_all_storage_path_columns(tmp_path: Path) -> None:
    """Images, nuclei, and annotations should move to canonical cluster roots.

    The source CSV remains unchanged while the output contains absolute paths
    that exist under training_images, test_images, and outputs.
    """
    data_root = tmp_path / "astroseg_data"
    image = data_root / "training_images" / "sample.tif"
    nucleus = data_root / "outputs" / "interim" / "nucleus_labels" / "sample.tiff"
    binary = data_root / "outputs" / "annotations" / "binary" / "sample.tiff"
    instance = data_root / "outputs" / "annotations" / "instances" / "sample.tiff"
    for path in (image, nucleus, binary, instance):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()

    row = {column: "" for column in MANIFEST_COLUMNS}
    row.update(
        {
            "image_id": "sample",
            "path": ".astroseg_runtime/dataset/training_images/sample.tif",
            "cellpose_mask_path": ".astroseg_runtime/outputs/interim/nucleus_labels/sample.tiff",
            "annotation_path": ".astroseg_runtime/outputs/annotations/binary/sample.tiff",
            "instance_annotation_path": (
                ".astroseg_runtime/outputs/annotations/instances/sample.tiff"
            ),
            "annotation_status": "seed",
            "split": "train",
        }
    )
    source = tmp_path / "manifest.csv"
    destination = tmp_path / "manifest_cluster.csv"
    pd.DataFrame([row], columns=MANIFEST_COLUMNS).to_csv(source, index=False)

    relocated = relocate_runtime_manifest(source, destination, data_root)

    assert destination.is_file()
    assert Path(relocated.loc[0, "path"]) == image.resolve()
    assert Path(relocated.loc[0, "cellpose_mask_path"]) == nucleus.resolve()
    assert Path(relocated.loc[0, "annotation_path"]) == binary.resolve()
    assert Path(relocated.loc[0, "instance_annotation_path"]) == instance.resolve()
    assert pd.read_csv(source).loc[0, "path"].startswith(".astroseg_runtime/")
