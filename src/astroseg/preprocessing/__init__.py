"""Model-input preprocessing helpers."""

from astroseg.preprocessing.channels import ChannelSelection, select_model_channels
from astroseg.preprocessing.distance_maps import create_nucleus_proximity_map
from astroseg.preprocessing.gfap_detection import GfapBootstrapResult, detect_gfap_bootstrap_mask
from astroseg.preprocessing.nucleus_detection import NucleusDetectionResult, detect_nucleus_instances
from astroseg.preprocessing.normalize import percentile_normalize
from astroseg.preprocessing.nuclei import labels_to_binary_mask, validate_nucleus_labels
from astroseg.preprocessing.patches import (
    PatchCoordinates,
    extract_patch,
    generate_patch_coordinates,
    stitch_probability_patches,
)

__all__ = [
    "ChannelSelection",
    "GfapBootstrapResult",
    "NucleusDetectionResult",
    "PatchCoordinates",
    "create_nucleus_proximity_map",
    "detect_gfap_bootstrap_mask",
    "detect_nucleus_instances",
    "extract_patch",
    "generate_patch_coordinates",
    "labels_to_binary_mask",
    "percentile_normalize",
    "select_model_channels",
    "stitch_probability_patches",
    "validate_nucleus_labels",
]
