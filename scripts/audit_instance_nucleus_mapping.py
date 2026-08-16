"""Audit how many annotated astrocytes have a unique nearby detected nucleus."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from astroseg.io import load_manifest
from astroseg.preprocessing import map_cells_to_nuclei


def _load_labels(path: str) -> np.ndarray:
    """Load one explicit TIFF/NumPy label array."""
    source = Path(path)
    return (
        np.load(source, allow_pickle=False)
        if source.suffix.lower() == ".npy"
        else tifffile.imread(source)
    )


def audit_mapping(manifest_path: Path, max_distance: float) -> pd.DataFrame:
    """Return per-image cell/nucleus counts and one-to-one mapping coverage."""
    manifest = load_manifest(manifest_path)
    selected = manifest.loc[
        manifest["instance_annotation_path"].astype(str).str.strip() != ""
    ]
    records = []
    for _, row in selected.iterrows():
        cells = _load_labels(str(row["instance_annotation_path"]))
        nuclei = _load_labels(str(row["cellpose_mask_path"]))
        mapping = map_cells_to_nuclei(cells, nuclei, max_distance)
        cell_count = int(np.unique(cells[cells > 0]).size)
        records.append(
            {
                "image_id": str(row["image_id"]),
                "cell_count": cell_count,
                "nucleus_count": int(np.unique(nuclei[nuclei > 0]).size),
                "mapped_cell_count": len(mapping),
                "mapped_cell_fraction": len(mapping) / cell_count,
            }
        )
    if not records:
        raise ValueError("Manifest contains no instance annotations")
    return pd.DataFrame(records)


def main() -> None:
    """Parse manifest/distance, save optional CSV, and print weighted coverage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=64)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    report = audit_mapping(args.manifest, args.max_distance)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        report.to_csv(args.output, index=False)
    coverage = report["mapped_cell_count"].sum() / report["cell_count"].sum()
    print(report.to_string(index=False))
    print(
        f"Mapped {report.mapped_cell_count.sum()}/{report.cell_count.sum()} "
        f"annotated cells ({coverage:.2%}) within {args.max_distance:g} px"
    )


if __name__ == "__main__":
    main()
