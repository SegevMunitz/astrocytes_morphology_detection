"""Patch and full-image inference utilities."""

from astroseg.inference.predict_full_image import predict_full_image
from astroseg.inference.predict_patch import predict_patch
from astroseg.inference.predict_instances import (
    InstanceHeadPredictions,
    predict_instance_full_image,
    predict_instance_patch,
)
from astroseg.inference.stitch import stitch_probability_patches

__all__ = [
    "InstanceHeadPredictions",
    "predict_full_image",
    "predict_instance_full_image",
    "predict_instance_patch",
    "predict_patch",
    "stitch_probability_patches",
]
