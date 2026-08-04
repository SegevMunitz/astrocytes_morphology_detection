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
    "instance_annotation_path",
    "compartment_annotation_path",
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
COMPARTMENT_CLASSES = {
    "background": 0,
    "nucleus": 1,
    "soma": 2,
    "process": 3,
}
