"""Matplotlib image previews, overlays, and QC montages."""

from astroseg.visualization.overlays import (
    save_compartment_overlay,
    save_instance_overlay,
    save_segmentation_overlay,
)
from astroseg.visualization.qc_plots import (
    save_gfap_preview,
    save_grayscale_preview,
    save_nucleus_label_preview,
    save_nucleus_mask_preview,
    save_proximity_map_preview,
    save_qc_montage,
)

__all__ = [
    "save_gfap_preview",
    "save_grayscale_preview",
    "save_nucleus_label_preview",
    "save_nucleus_mask_preview",
    "save_proximity_map_preview",
    "save_qc_montage",
    "save_segmentation_overlay",
    "save_instance_overlay",
    "save_compartment_overlay",
]
