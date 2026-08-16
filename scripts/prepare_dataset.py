"""Automatically select channels, detect nuclei, create QC, and update a manifest."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from astroseg.io import get_channel, load_manifest, load_ome_tiff, validate_manifest
from astroseg.preprocessing import (
    create_nucleus_proximity_map,
    detect_nucleus_instances,
    prepare_dapi_for_detection,
    select_model_channels,
)
from astroseg.visualization import (
    save_gfap_preview,
    save_grayscale_preview,
    save_nucleus_label_preview,
    save_nucleus_mask_preview,
    save_proximity_map_preview,
    save_qc_montage,
)


def _resolve_path(value: str, manifest_path: Path) -> Path:
    """Resolve a source image path using project- and manifest-relative layouts.

    The original value is included in failures so batch preparation identifies
    the exact manifest entry requiring attention.
    """
    path = Path(value)
    for candidate in (path, manifest_path.parent / path):
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"Manifest image path does not exist: {value}")


def _portable_path(path: Path) -> str:
    """Store generated paths relative to the project whenever possible.

    Forward slashes keep generated manifests portable while external destinations
    remain explicit absolute or caller-supplied paths.
    """
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def prepare_dataset(
    manifest_path: Path,
    output_directory: Path,
    output_manifest: Path | None = None,
    gaussian_sigma: float = 1.2,
    threshold_scale: float = 1.0,
    min_nucleus_area: int = 30,
    min_peak_distance: int = 7,
    max_nucleus_distance: float = 64.0,
) -> pd.DataFrame:
    """Prepare every manifest image without per-image channel or nucleus work.

    The function selects channels, writes extracted arrays and previews, detects
    nucleus instances, derives model inputs, creates QC, and updates nucleus paths.
    """
    manifest = load_manifest(manifest_path)
    destination_manifest = output_manifest or manifest_path
    channel_directory = output_directory / "channels"
    channel_preview_directory = channel_directory / "previews"
    label_directory = output_directory / "nucleus_labels"
    mask_directory = output_directory / "nucleus_masks"
    distance_directory = output_directory / "nucleus_distance_maps"
    qc_directory = output_directory / "qc"
    records: list[dict[str, object]] = []

    for index, row in manifest.iterrows():
        image_id = str(row["image_id"])
        microscopy = load_ome_tiff(_resolve_path(str(row["path"]), manifest_path))
        selection = select_model_channels(
            microscopy,
            str(row["gfap_channel"]),
            str(row["dapi_channel"]),
            str(row.get("auxiliary_channel", "")),
        )
        gfap = get_channel(microscopy, selection.gfap_channel)
        dapi = get_channel(microscopy, selection.dapi_channel)
        detection_dapi, dapi_preprocessing = prepare_dapi_for_detection(
            dapi, gfap, selection
        )
        detection = detect_nucleus_instances(
            detection_dapi,
            gaussian_sigma=gaussian_sigma,
            threshold_scale=threshold_scale,
            min_nucleus_area=min_nucleus_area,
            min_peak_distance=min_peak_distance,
        )
        proximity = create_nucleus_proximity_map(
            detection.binary_mask, max_distance=max_nucleus_distance
        )

        for directory in (
            channel_directory,
            channel_preview_directory,
            label_directory,
            mask_directory,
            distance_directory,
            qc_directory,
        ):
            directory.mkdir(parents=True, exist_ok=True)
        np.save(channel_directory / f"{image_id}_gfap.npy", gfap, allow_pickle=False)
        np.save(channel_directory / f"{image_id}_dapi.npy", dapi, allow_pickle=False)
        save_gfap_preview(gfap, channel_preview_directory / f"{image_id}_gfap.png")
        save_grayscale_preview(
            dapi, channel_preview_directory / f"{image_id}_dapi.png", "DAPI"
        )

        label_path = label_directory / f"{image_id}_nuclei.tiff"
        tifffile.imwrite(label_path, detection.labels, photometric="minisblack")
        np.save(mask_directory / f"{image_id}.npy", detection.binary_mask, allow_pickle=False)
        np.save(distance_directory / f"{image_id}.npy", proximity, allow_pickle=False)
        save_nucleus_label_preview(
            detection.labels, qc_directory / f"{image_id}_labels.png"
        )
        save_nucleus_mask_preview(
            detection.binary_mask, qc_directory / f"{image_id}_mask.png"
        )
        save_proximity_map_preview(
            proximity, qc_directory / f"{image_id}_proximity.png"
        )
        save_qc_montage(
            gfap,
            detection.labels,
            detection.binary_mask,
            proximity,
            qc_directory / f"{image_id}_montage.png",
            dapi=dapi,
        )

        manifest.loc[index, "gfap_channel"] = selection.gfap_channel
        manifest.loc[index, "auxiliary_channel"] = selection.auxiliary_channel
        manifest.loc[index, "dapi_channel"] = selection.dapi_channel
        manifest.loc[index, "cellpose_mask_path"] = _portable_path(label_path)
        records.append(
            {
                "image_id": image_id,
                "gfap_channel": selection.gfap_channel,
                "auxiliary_channel": selection.auxiliary_channel,
                "dapi_channel": selection.dapi_channel,
                "channel_selection_method": selection.method,
                "dapi_preprocessing": dapi_preprocessing,
                "red_score": selection.red_score,
                "green_score": selection.green_score,
                "nucleus_threshold": detection.threshold,
                "nucleus_count": detection.instance_count,
                "nucleus_foreground_fraction": detection.foreground_fraction,
                "nucleus_label_path": _portable_path(label_path),
                "gaussian_sigma": gaussian_sigma,
                "threshold_scale": threshold_scale,
                "min_nucleus_area": min_nucleus_area,
                "min_peak_distance": min_peak_distance,
                "max_nucleus_distance": max_nucleus_distance,
            }
        )

    validate_manifest(manifest)
    destination_manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(destination_manifest, index=False)
    pd.DataFrame(records).to_csv(qc_directory / "preparation_report.csv", index=False)
    return manifest


def parse_args() -> argparse.Namespace:
    """Parse automatic preparation paths and nucleus-detector parameters.

    The input manifest is updated by default; an explicit output path can create a
    separate prepared manifest when versioned metadata is preferred.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=Path("data/metadata/manifest.csv")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("data/interim"))
    parser.add_argument("--output-manifest", type=Path)
    parser.add_argument("--gaussian-sigma", type=float, default=1.2)
    parser.add_argument("--threshold-scale", type=float, default=1.0)
    parser.add_argument("--min-nucleus-area", type=int, default=30)
    parser.add_argument("--min-peak-distance", type=int, default=7)
    parser.add_argument("--max-nucleus-distance", type=float, default=64.0)
    return parser.parse_args()


def main() -> None:
    """Run automatic preparation and report the completed image count.

    Derived files are reproducible and may be regenerated; raw images and human
    annotations are never modified by this command.
    """
    args = parse_args()
    manifest = prepare_dataset(
        args.manifest,
        args.output_dir,
        args.output_manifest,
        args.gaussian_sigma,
        args.threshold_scale,
        args.min_nucleus_area,
        args.min_peak_distance,
        args.max_nucleus_distance,
    )
    destination = args.output_manifest or args.manifest
    print(f"Prepared {len(manifest)} images and updated {destination}")


if __name__ == "__main__":
    main()
