"""Non-destructive import of manually corrected instance or binary masks."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import shutil

import numpy as np
import tifffile

from astroseg.constants import HUMAN_ANNOTATION_STATUSES
from astroseg.io.ome_tiff import MicroscopyImage, get_channel, load_ome_tiff
from astroseg.visualization.overlays import save_segmentation_overlay


@dataclass(frozen=True)
class AnnotationImportResult:
    """Artifacts and provenance produced by one annotation import.

    The immutable record identifies the preserved source mask, derived binary
    target, QC overlay, and lifecycle metadata written back to the manifest.
    """

    image_id: str
    original_mask_path: Path
    binary_mask_path: Path
    qc_overlay_path: Path
    annotation_status: str
    annotation_source: str
    annotator: str
    review_status: str


def load_annotation_mask(path: str | Path) -> np.ndarray:
    """Load a two-dimensional annotation mask without modifying its source.

    NumPy and TIFF masks are supported, and their original numeric dtype is
    retained. Shape and label semantics are validated by the import workflow.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Annotation mask does not exist: {source}")
    if source.suffix.lower() == ".npy":
        mask = np.load(source, allow_pickle=False)
    elif source.suffix.lower() in {".tif", ".tiff"}:
        mask = tifffile.imread(source)
    else:
        raise ValueError(f"Unsupported annotation-mask format {source.suffix!r}: {source}")
    mask = np.asarray(mask)
    if mask.ndim != 2:
        raise ValueError(f"Annotation mask must be 2D; received shape {mask.shape} from {source}")
    return mask


def validate_annotation_alignment(microscopy_image: MicroscopyImage, mask: np.ndarray) -> None:
    """Validate pixel-grid alignment using exact dimensions and label integrity.

    Shape equality verifies the available structural alignment information. A
    visual QC overlay is still required because TIFF files may not contain enough
    spatial metadata to prove biological registration.
    """
    image_shape = microscopy_image.image.shape[-2:]
    if mask.ndim != 2 or mask.shape != image_shape:
        raise ValueError(f"Annotation shape {mask.shape} does not match image shape {image_shape}")
    if not np.issubdtype(mask.dtype, np.number):
        raise TypeError("Annotation mask must have a numeric dtype")
    if not np.isfinite(mask).all():
        raise ValueError("Annotation mask contains non-finite values")
    if np.any(mask < 0):
        raise ValueError("Annotation mask must not contain negative labels")
    if np.issubdtype(mask.dtype, np.floating) and not np.equal(mask, np.floor(mask)).all():
        raise ValueError("Annotation mask must contain integer-valued labels")


def _prepare_destination(path: Path, overwrite: bool) -> None:
    """Validate overwrite policy and create the destination parent directory.

    Existing artifacts are protected by default so annotation imports cannot
    silently replace human work. No file content is written by this helper.
    """
    if path.exists() and not overwrite:
        raise FileExistsError(f"Refusing to overwrite existing annotation artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)


def import_annotation_pair(
    image_id: str,
    image_path: str | Path,
    mask_path: str | Path,
    gfap_channel: str,
    output_directory: str | Path,
    annotation_status: str = "seed",
    annotation_source: str = "manual_cellpose_correction",
    annotator: str = "",
    review_status: str = "",
    overwrite: bool = False,
) -> AnnotationImportResult:
    """Import one human mask without overwriting the original export.

    The function validates image alignment, archives the instance-valued mask by
    content hash, derives a binary target, and writes a GFAP QC overlay.
    """
    if not image_id.strip():
        raise ValueError("image_id must not be empty")
    status = annotation_status.strip().lower()
    if status not in HUMAN_ANNOTATION_STATUSES:
        raise ValueError(
            f"Imported human annotations require one of {sorted(HUMAN_ANNOTATION_STATUSES)}; got {status!r}"
        )
    if not annotation_source.strip():
        raise ValueError("annotation_source must not be empty")
    source_mask = Path(mask_path)
    microscopy = load_ome_tiff(image_path)
    mask = load_annotation_mask(source_mask)
    validate_annotation_alignment(microscopy, mask)
    gfap = get_channel(microscopy, gfap_channel)

    destination = Path(output_directory)
    digest = hashlib.sha256(source_mask.read_bytes()).hexdigest()[:12]
    original_path = (
        destination / "originals" / image_id / f"{status}_{digest}{source_mask.suffix.lower()}"
    )
    binary_path = destination / "binary" / f"{image_id}_{status}_binary.tiff"
    overlay_path = destination / "qc" / f"{image_id}_{status}_annotation_overlay.png"
    for path in (binary_path, overlay_path):
        _prepare_destination(path, overwrite)
    original_path.parent.mkdir(parents=True, exist_ok=True)
    if not original_path.exists():
        shutil.copy2(source_mask, original_path)
    binary = (mask > 0).astype(np.uint8)
    tifffile.imwrite(binary_path, binary)
    save_segmentation_overlay(gfap, binary, overlay_path, title="Imported annotation")
    return AnnotationImportResult(
        image_id=image_id,
        original_mask_path=original_path,
        binary_mask_path=binary_path,
        qc_overlay_path=overlay_path,
        annotation_status=status,
        annotation_source=annotation_source.strip(),
        annotator=annotator.strip(),
        review_status=review_status.strip(),
    )
