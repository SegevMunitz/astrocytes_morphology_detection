"""Non-destructive import of complete-cell astrocyte instance annotations."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import numpy as np

from astroseg.annotations.workflow import (
    AnnotationImportResult,
    import_annotation_pair,
    load_annotation_mask,
    validate_annotation_alignment,
)
from astroseg.io import get_channel, load_ome_tiff
from astroseg.preprocessing import build_astrocyte_instance_targets
from astroseg.visualization import save_compartment_overlay, save_instance_overlay


@dataclass(frozen=True)
class InstanceAnnotationImportResult:
    """Preserved instance data, derived binary mask, QC, and provenance.

    The archived full-cell instance mask is the authoritative training target.
    Optional compartment labels are preserved separately without relabeling.
    """

    base: AnnotationImportResult
    instance_mask_path: Path
    compartment_mask_path: Path | None
    instance_overlay_path: Path
    compartment_overlay_path: Path | None


def import_astrocyte_instance_pair(
    image_id: str,
    image_path: str | Path,
    nucleus_mask_path: str | Path,
    instance_mask_path: str | Path,
    gfap_channel: str,
    output_directory: str | Path,
    compartment_mask_path: str | Path | None = None,
    annotation_status: str = "seed",
    annotation_source: str = "manual_complete_cell_correction",
    annotator: str = "",
    review_status: str = "",
    soma_radius: float = 20.0,
    offset_scale: float = 256.0,
    overwrite: bool = False,
) -> InstanceAnnotationImportResult:
    """Validate and archive one ownership-aware complete-cell annotation.

    Dimensions, labels, one-cell/one-nucleus mapping, and optional compartments are
    checked before writes. The existing binary import remains as a derived target.
    """
    microscopy = load_ome_tiff(image_path)
    instances = load_annotation_mask(instance_mask_path)
    nuclei = load_annotation_mask(nucleus_mask_path)
    compartments = (
        load_annotation_mask(compartment_mask_path) if compartment_mask_path is not None else None
    )
    validate_annotation_alignment(microscopy, instances)
    validate_annotation_alignment(microscopy, nuclei)
    if compartments is not None:
        validate_annotation_alignment(microscopy, compartments)
    build_astrocyte_instance_targets(
        instances,
        nuclei,
        compartments,
        soma_radius=soma_radius,
        offset_scale=offset_scale,
    )

    destination = Path(output_directory)
    status = annotation_status.strip().lower()
    instance_overlay = destination / "qc" / f"{image_id}_{status}_instance_ids.png"
    compartment_overlay = (
        destination / "qc" / f"{image_id}_{status}_compartments.png"
        if compartments is not None
        else None
    )
    extra_destinations = [instance_overlay]
    if compartment_overlay is not None:
        extra_destinations.append(compartment_overlay)
    existing = [path for path in extra_destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"Refusing to overwrite instance annotation QC: {existing}")

    base = import_annotation_pair(
        image_id,
        image_path,
        instance_mask_path,
        gfap_channel,
        output_directory,
        annotation_status,
        annotation_source,
        annotator,
        review_status,
        overwrite,
    )
    gfap = get_channel(microscopy, gfap_channel)
    instance_overlay.parent.mkdir(parents=True, exist_ok=True)
    save_instance_overlay(gfap, instances, instance_overlay)

    archived_compartment: Path | None = None
    if compartment_mask_path is not None and compartments is not None:
        source = Path(compartment_mask_path)
        digest = hashlib.sha256(source.read_bytes()).hexdigest()[:12]
        archived_compartment = (
            destination
            / "compartment_originals"
            / image_id
            / f"{status}_{digest}{source.suffix.lower()}"
        )
        archived_compartment.parent.mkdir(parents=True, exist_ok=True)
        if not archived_compartment.exists():
            shutil.copy2(source, archived_compartment)
        assert compartment_overlay is not None
        save_compartment_overlay(gfap, compartments, compartment_overlay)
    return InstanceAnnotationImportResult(
        base=base,
        instance_mask_path=base.original_mask_path,
        compartment_mask_path=archived_compartment,
        instance_overlay_path=instance_overlay,
        compartment_overlay_path=compartment_overlay,
    )
