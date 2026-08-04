"""Small dependency-free augmentations for aligned segmentation arrays."""

import random
from dataclasses import dataclass

import numpy as np


@dataclass
class RandomFlip:
    """Randomly flip an aligned input-target pair along spatial axes.

    Horizontal and vertical decisions are sampled independently, and every
    transform is applied identically to all image channels and the target.
    """

    horizontal_probability: float = 0.5
    vertical_probability: float = 0.5

    def __post_init__(self) -> None:
        """Validate horizontal and vertical flip probabilities after construction.

        Each probability must lie within the closed interval from zero to one;
        invalid augmentation configuration fails before dataset iteration.
        """
        for value in (self.horizontal_probability, self.vertical_probability):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Flip probabilities must be in [0, 1]")

    def __call__(self, image: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Apply sampled flips and return aligned contiguous arrays.

        Contiguous copies avoid negative NumPy strides, which PyTorch cannot
        convert directly after a ``numpy.flip`` operation.
        """
        if random.random() < self.horizontal_probability:
            image = np.flip(image, axis=-1)
            target = np.flip(target, axis=-1)
        if random.random() < self.vertical_probability:
            image = np.flip(image, axis=-2)
            target = np.flip(target, axis=-2)
        return np.ascontiguousarray(image), np.ascontiguousarray(target)


@dataclass
class RandomInstanceFlip(RandomFlip):
    """Flip image and all instance targets while correcting vector directions.

    Spatial masks receive identical transforms. Horizontal flips negate ``dx``
    offsets and vertical flips negate ``dy`` so ownership vectors remain valid.
    """

    def __call__(
        self,
        image: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Apply aligned random flips to multi-head instance training arrays.

        Every returned array is contiguous, avoiding negative strides during
        conversion to PyTorch tensors.
        """
        transformed = {name: np.asarray(value) for name, value in targets.items()}
        if random.random() < self.horizontal_probability:
            image = np.flip(image, axis=-1)
            transformed = {name: np.flip(value, axis=-1) for name, value in transformed.items()}
            transformed["offsets"] = transformed["offsets"].copy()
            transformed["offsets"][1] *= -1
        if random.random() < self.vertical_probability:
            image = np.flip(image, axis=-2)
            transformed = {name: np.flip(value, axis=-2) for name, value in transformed.items()}
            transformed["offsets"] = transformed["offsets"].copy()
            transformed["offsets"][0] *= -1
        return np.ascontiguousarray(image), {
            name: np.ascontiguousarray(value) for name, value in transformed.items()
        }
