"""Separate complete astrocytes using GFAP foreground and nucleus ownership seeds."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from astroseg.analysis import astrocyte_instance_morphology
from astroseg.io import get_channel, load_manifest, load_ome_tiff, validate_manifest
from astroseg.postprocessing import separate_astrocyte_instances
from astroseg.visualization import save_compartment_overlay, save_instance_overlay


def _resolve_path(value: str, manifest_path: Path, description: str) -> Path:
    """Resolve project- or manifest-relative input with a descriptive failure.

    Exact paths remain mandatory; the script does not guess masks by similar names.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"{description} does not exist: {value}")


def _portable_path(path: Path) -> str:
    """Represent generated paths relative to the project where possible.

    Forward slashes keep the output manifest portable across supported platforms.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _load_array(path: Path) -> np.ndarray:
    """Load a two-dimensional TIFF/NumPy label image without reinterpretation.

    Shape validation occurs against microscopy dimensions in the main workflow.
    """
    array = np.load(path, allow_pickle=False) if path.suffix.lower() == ".npy" else tifffile.imread(path)
    return np.asarray(array)


def generate_astrocyte_instances(
    manifest_path: Path,
    output_directory: Path,
    output_manifest: Path,
    probability_directory: Path | None = None,
    foreground_mask_directory: Path | None = None,
    overwrite: bool = False,
    foreground_threshold: float = 0.5,
    max_nucleus_to_gfap_distance: float = 16.0,
    soma_expansion: int = 4,
    soma_radius: float = 20.0,
    min_cell_area: int = 50,
    min_gfap_area: int = 20,
    nucleus_support_expansion: int = 4,
    min_nucleus_foreground_fraction: float = 0.0,
) -> pd.DataFrame:
    """Generate nucleus-owned bootstrap instances for every manifest image.

    Automatic outputs and their manifest are separate from human instance
    annotations. Learned offsets are added by ``predict_astrocyte_instances.py``;
    this pre-checkpoint workflow uses boundary-aware watershed as an explicit
    pseudo-label bootstrap.
    """
    manifest = load_manifest(manifest_path)
    labels_directory = output_directory / "labels"
    compartments_directory = output_directory / "compartments"
    nuclei_directory = output_directory / "nucleus_cell_labels"
    overlays_directory = output_directory / "overlays"
    cell_records: list[pd.DataFrame] = []
    image_records: list[dict[str, object]] = []

    for index, row in manifest.iterrows():
        image_id = str(row["image_id"])
        if foreground_mask_directory is None:
            mask_value = str(row["annotation_path"]).strip()
            if not mask_value:
                raise ValueError(
                    f"No foreground mask is available for {image_id!r}; provide --foreground-mask-dir"
                )
            mask_path = _resolve_path(mask_value, manifest_path, "Foreground mask")
        else:
            mask_path = _resolve_path(
                str(foreground_mask_directory / f"{image_id}.tiff"),
                manifest_path,
                "Foreground mask",
            )
        nucleus_path = _resolve_path(
            str(row["cellpose_mask_path"]), manifest_path, "Nucleus instance labels"
        )
        microscopy = load_ome_tiff(_resolve_path(str(row["path"]), manifest_path, "Image"))
        gfap = get_channel(microscopy, str(row["gfap_channel"]))
        foreground_mask = _load_array(mask_path)
        nuclei = _load_array(nucleus_path)
        if foreground_mask.shape != gfap.shape or nuclei.shape != gfap.shape:
            raise ValueError(f"Image, foreground, and nuclei are not aligned for {image_id!r}")

        cell_probability = (foreground_mask > 0).astype(np.float32)
        if probability_directory is not None:
            probability_path = probability_directory / f"{image_id}.npy"
            if probability_path.is_file():
                probabilities = np.load(probability_path, allow_pickle=False)
                if probabilities.shape != (2, *gfap.shape):
                    raise ValueError(
                        f"Foreground probabilities for {image_id!r} must have shape [2, H, W]"
                    )
                cell_probability = probabilities[1].astype(np.float32, copy=False)

        result = separate_astrocyte_instances(
            cell_probability,
            nuclei,
            foreground_threshold=foreground_threshold,
            max_nucleus_to_gfap_distance=max_nucleus_to_gfap_distance,
            nucleus_support_expansion=nucleus_support_expansion,
            min_nucleus_foreground_fraction=min_nucleus_foreground_fraction,
            soma_expansion=soma_expansion,
            soma_radius=soma_radius,
            min_cell_area=min_cell_area,
            min_gfap_area=min_gfap_area,
        )
        label_path = labels_directory / f"{image_id}.tiff"
        compartment_path = compartments_directory / f"{image_id}.tiff"
        nucleus_cell_path = nuclei_directory / f"{image_id}.tiff"
        instance_overlay_path = overlays_directory / f"{image_id}_instances.png"
        compartment_overlay_path = overlays_directory / f"{image_id}_compartments.png"
        destinations = (
            label_path,
            compartment_path,
            nucleus_cell_path,
            instance_overlay_path,
            compartment_overlay_path,
        )
        existing = [path for path in destinations if path.exists()]
        if existing and not overwrite:
            raise FileExistsError(f"Refusing to overwrite automatic instance artifacts: {existing}")
        for path in destinations:
            path.parent.mkdir(parents=True, exist_ok=True)
        tifffile.imwrite(label_path, result.labels, photometric="minisblack")
        tifffile.imwrite(compartment_path, result.compartments, photometric="minisblack")
        tifffile.imwrite(nucleus_cell_path, result.nucleus_cell_labels, photometric="minisblack")
        save_instance_overlay(gfap, result.labels, instance_overlay_path)
        save_compartment_overlay(gfap, result.compartments, compartment_overlay_path)

        measurements = astrocyte_instance_morphology(
            result.labels, result.compartments, result.cell_to_nucleus
        )
        measurements.insert(0, "image_id", image_id)
        cell_records.append(measurements)
        manifest.loc[index, "predicted_instance_path"] = _portable_path(label_path)
        manifest.loc[index, "predicted_compartment_path"] = _portable_path(compartment_path)
        manifest.loc[index, "instance_prediction_status"] = "pseudo"
        manifest.loc[index, "instance_prediction_source"] = result.ownership_mode
        image_records.append(
            {
                "image_id": image_id,
                "cell_count": result.cell_count,
                "active_nucleus_count": result.active_nucleus_count,
                "rejected_nucleus_count": result.rejected_nucleus_count,
                "unassigned_foreground_fraction": result.unassigned_foreground_fraction,
                "ownership_mode": result.ownership_mode,
                "instance_path": _portable_path(label_path),
                "compartment_path": _portable_path(compartment_path),
            }
        )

    validate_manifest(manifest)
    output_directory.mkdir(parents=True, exist_ok=True)
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_manifest, index=False)
    pd.concat(cell_records, ignore_index=True).to_csv(
        output_directory / "cell_measurements.csv", index=False
    )
    pd.DataFrame(image_records).to_csv(output_directory / "instance_report.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse bootstrap instance paths and nucleus-assignment parameters.

    Defaults consume the automatic GFAP pseudo manifest and preserve all outputs
    below a dedicated astrocyte-instance directory.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("outputs/pseudo_labels/manifest.csv")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/astrocyte_instances")
    )
    parser.add_argument(
        "--output-manifest",
        type=Path,
        default=Path("outputs/astrocyte_instances/manifest.csv"),
    )
    parser.add_argument(
        "--probability-dir", type=Path, default=Path("outputs/pseudo_labels/probabilities")
    )
    parser.add_argument("--foreground-mask-dir", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--foreground-threshold", type=float, default=0.5)
    parser.add_argument("--max-nucleus-to-gfap-distance", type=float, default=16.0)
    parser.add_argument("--nucleus-support-expansion", type=int, default=4)
    parser.add_argument(
        "--min-nucleus-foreground-fraction", type=float, default=0.0
    )
    parser.add_argument("--soma-expansion", type=int, default=4)
    parser.add_argument("--soma-radius", type=float, default=20.0)
    parser.add_argument("--min-cell-area", type=int, default=50)
    parser.add_argument("--min-gfap-area", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    """Run bootstrap separation and report the total generated cell count.

    The source pseudo manifest and every human annotation remain unchanged.
    """
    args = parse_args()
    generate_astrocyte_instances(
        args.manifest,
        args.output_dir,
        args.output_manifest,
        args.probability_dir,
        args.foreground_mask_dir,
        args.overwrite,
        args.foreground_threshold,
        args.max_nucleus_to_gfap_distance,
        args.soma_expansion,
        args.soma_radius,
        args.min_cell_area,
        args.min_gfap_area,
        args.nucleus_support_expansion,
        args.min_nucleus_foreground_fraction,
    )
    report = pd.read_csv(args.output_dir / "instance_report.csv")
    print(
        f"Separated {int(report['cell_count'].sum())} astrocyte instances "
        f"across {len(report)} image(s)"
    )


if __name__ == "__main__":
    main()
