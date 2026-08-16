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


@dataclass
class RandomInstanceAugmentation(RandomInstanceFlip):
    """Apply dihedral geometry and conservative fluorescence perturbations."""

    rotation_probability: float = 0.75
    intensity_probability: float = 0.8
    noise_standard_deviation: float = 0.02
    auxiliary_dropout_probability: float = 0.0

    def __post_init__(self) -> None:
        super().__post_init__()
        for value in (
            self.rotation_probability,
            self.intensity_probability,
            self.auxiliary_dropout_probability,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("Augmentation probabilities must be in [0, 1]")
        if self.noise_standard_deviation < 0:
            raise ValueError("noise_standard_deviation must be non-negative")

    def __call__(
        self,
        image: np.ndarray,
        targets: dict[str, np.ndarray],
    ) -> tuple[np.ndarray, dict[str, np.ndarray]]:
        """Transform every target and rotate ownership vectors analytically."""
        transformed = {name: np.asarray(value) for name, value in targets.items()}
        if random.random() < self.rotation_probability:
            rotations = random.randint(1, 3)
            image = np.rot90(image, rotations, axes=(-2, -1))
            transformed = {
                name: np.rot90(value, rotations, axes=(-2, -1))
                for name, value in transformed.items()
            }
            offsets = transformed["offsets"].copy()
            for _ in range(rotations):
                dy, dx = offsets[0].copy(), offsets[1].copy()
                offsets[0], offsets[1] = -dx, dy
            transformed["offsets"] = offsets

        image, transformed = super().__call__(image, transformed)
        if random.random() < self.auxiliary_dropout_probability:
            if image.shape[0] != 3:
                raise ValueError("Auxiliary-channel dropout requires exactly three inputs")
            image = image.copy()
            image[1] = 0
        if random.random() < self.intensity_probability:
            perturbed = image.astype(np.float32, copy=True)
            for channel in range(perturbed.shape[0]):
                # Preserve the explicit all-zero placeholder used when GFP was
                # not acquired; adding noise would fabricate an auxiliary stain.
                if not np.any(perturbed[channel]):
                    continue
                scale = random.uniform(0.8, 1.2)
                gamma = random.uniform(0.85, 1.15)
                plane = np.clip(perturbed[channel] * scale, 0.0, 1.0) ** gamma
                if self.noise_standard_deviation:
                    plane += np.random.normal(
                        0.0, self.noise_standard_deviation, size=plane.shape
                    ).astype(np.float32)
                perturbed[channel] = np.clip(plane, 0.0, 1.0)
            image = perturbed
        return np.ascontiguousarray(image), {
            name: np.ascontiguousarray(value) for name, value in transformed.items()
        }
