"""Shared schema and class constants."""

MANIFEST_COLUMNS = [
    "image_id",
    "experiment_id",
    "timepoint",
    "treatment",
    "magnification",
    "path",
    "gfap_channel",
    "dapi_channel",
    "cellpose_mask_path",
    "annotation_path",
    "split",
]

VALID_SPLITS = {"", "train", "val", "test"}
BACKGROUND_CLASS = 0

