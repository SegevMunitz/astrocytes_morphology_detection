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
    "annotation_status",
    "annotation_source",
    "annotator",
    "review_status",
    "split",
]

VALID_SPLITS = {"", "train", "val", "test"}
ANNOTATION_STATUSES = {"none", "seed", "pseudo", "corrected", "reviewed"}
TRAINABLE_ANNOTATION_STATUSES = frozenset({"seed", "corrected", "reviewed"})
HUMAN_ANNOTATION_STATUSES = frozenset({"seed", "corrected", "reviewed"})
BACKGROUND_CLASS = 0
