"""Desktop mask-review helpers and the Cellpose-like comparison GUI."""

from astroseg.gui.mask_review import (
    MaskReviewDataset,
    SavedCorrection,
    load_instance_mask,
    paint_instance_disk,
    render_instance_overlay,
    save_corrected_instances,
)

__all__ = [
    "MaskReviewDataset",
    "SavedCorrection",
    "load_instance_mask",
    "paint_instance_disk",
    "render_instance_overlay",
    "save_corrected_instances",
]
