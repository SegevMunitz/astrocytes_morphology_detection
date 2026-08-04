"""Microscopy TIFF/BMP loading with explicit channel-axis handling."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tifffile
from ome_types import from_xml
from PIL import Image

from astroseg.constants import MICROSCOPY_IMAGE_SUFFIXES


@dataclass
class MicroscopyImage:
    """A microscopy image with a safe channel-first representation.

    The record retains original numeric values, explicit OME channel names,
    optional lateral pixel size, and the source path used for traceability.
    """

    image: np.ndarray
    channel_names: list[str]
    pixel_size_um: float | None
    source_path: Path


def _normalize_axes(array: np.ndarray, axes: str, source: Path) -> np.ndarray:
    """Convert a TIFF series with explicit axes to channel-first layout.

    Singleton non-spatial planes may be removed, but non-singleton time or depth
    axes are rejected so the loader never chooses a scientific plane implicitly.
    """
    axes = axes.upper()
    if len(axes) != array.ndim or "Y" not in axes or "X" not in axes:
        raise ValueError(
            f"Cannot safely interpret {source}: shape {array.shape} has axes {axes!r}. "
            "Expected explicit X and Y axes."
        )

    data = array
    current_axes = list(axes)
    for index in range(len(current_axes) - 1, -1, -1):
        axis = current_axes[index]
        if axis in {"C", "S", "Y", "X"}:
            continue
        if data.shape[index] != 1:
            raise ValueError(
                f"Cannot safely collapse non-singleton {axis!r} axis in {source}; "
                f"shape={array.shape}, axes={axes!r}. Select a plane before loading."
            )
        data = np.squeeze(data, axis=index)
        current_axes.pop(index)

    if "C" in current_axes and "S" in current_axes:
        raise ValueError(f"Both C and S axes are present in {source}; channel layout is ambiguous.")
    channel_axis_name = "C" if "C" in current_axes else ("S" if "S" in current_axes else None)
    if channel_axis_name is None:
        if current_axes != ["Y", "X"]:
            raise ValueError(f"Cannot interpret axes {''.join(current_axes)!r} in {source}.")
        return data[np.newaxis, ...]

    channel_axis = current_axes.index(channel_axis_name)
    y_axis = current_axes.index("Y")
    x_axis = current_axes.index("X")
    if len(current_axes) != 3:
        raise ValueError(f"Expected exactly channel, Y, and X axes in {source}; got {current_axes}.")
    return np.moveaxis(data, (channel_axis, y_axis, x_axis), (0, 1, 2))


def _parse_ome_metadata(xml: str | None, channel_count: int) -> tuple[list[str], float | None]:
    """Extract explicit channel names and lateral pixel size from OME XML.

    Missing metadata yields empty channel names and no pixel size rather than
    inferred identities. Metadata channel count must match the normalized array.
    """
    names = [""] * channel_count
    pixel_size: float | None = None
    if not xml:
        return names, pixel_size

    ome = from_xml(xml)
    if not ome.images:
        return names, pixel_size
    pixels = ome.images[0].pixels
    parsed_names = [channel.name or "" for channel in pixels.channels]
    if len(parsed_names) == channel_count:
        names = parsed_names

    if pixels.physical_size_x is not None:
        value = float(pixels.physical_size_x)
        unit = str(pixels.physical_size_x_unit or "um").lower()
        if "nano" in unit or unit.endswith("nm"):
            value /= 1000.0
        elif "milli" in unit or unit.endswith("mm"):
            value *= 1000.0
        pixel_size = value
    return names, pixel_size


def _load_bitmap(source: Path) -> MicroscopyImage:
    """Load a standard grayscale, RGB, or RGBA BMP as channel-first data.

    BMP contains no OME channel names or physical pixel size. Color samples are
    therefore exposed explicitly as Red, Green, Blue, and optional Alpha.
    """
    # Pillow identifies the file from its header, so a valid BMP remains readable
    # even when an acquisition/export tool gave it a misleading ``.tif`` suffix.
    with Image.open(source) as bitmap:
        array = np.asarray(bitmap).copy()
    if array.ndim == 2:
        image = array[np.newaxis, ...]
        channel_names = ["Gray"]
    elif array.ndim == 3 and array.shape[-1] in {3, 4}:
        image = np.moveaxis(array, -1, 0)
        channel_names = ["Red", "Green", "Blue"]
        if image.shape[0] == 4:
            channel_names.append("Alpha")
    else:
        raise ValueError(
            f"BMP must be grayscale, RGB, or RGBA; received shape {array.shape} from {source}"
        )
    return MicroscopyImage(image, channel_names, None, source)


def _detect_container(source: Path) -> str:
    """Identify BMP versus TIFF from magic bytes instead of trusting the suffix.

    Some microscopy exports keep a TIFF filename while writing BMP bytes. Content
    detection makes these files usable while unsupported or corrupt data still
    fail with a clear error before an image decoder is selected.
    """
    with source.open("rb") as handle:
        signature = handle.read(4)
    if signature[:2] == b"BM":
        return "bmp"
    if signature in {b"II*\x00", b"MM\x00*", b"II+\x00", b"MM\x00+"}:
        return "tiff"
    raise ValueError(
        f"Unsupported or corrupt microscopy image container in {source}; "
        "expected BMP or TIFF bytes"
    )


def load_microscopy_image(path: str | Path) -> MicroscopyImage:
    """Load one supported TIFF/OME-TIFF or BMP microscopy image.

    Singleton time and depth axes are removed. Non-singleton time/depth axes are
    rejected for TIFF because choosing a plane implicitly would be unsafe. RGB
    TIFF and BMP files receive explicit Red, Green, and Blue channel names.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Microscopy image does not exist: {source}")
    suffix = source.suffix.lower()
    if suffix not in MICROSCOPY_IMAGE_SUFFIXES:
        raise ValueError(
            f"Unsupported microscopy format {source.suffix!r}; expected BMP, TIF, or TIFF"
        )
    if _detect_container(source) == "bmp":
        return _load_bitmap(source)
    with tifffile.TiffFile(source) as tif:
        if not tif.series:
            raise ValueError(f"TIFF contains no image series: {source}")
        series = tif.series[0]
        array = series.asarray()
        image = _normalize_axes(array, series.axes, source)
        channel_names, pixel_size = _parse_ome_metadata(tif.ome_metadata, image.shape[0])
        photometric = series.pages[0].photometric.name if len(series.pages) else ""
        if (
            not any(channel_names)
            and series.axes.upper() == "YXS"
            and image.shape[0] in {3, 4}
            and photometric == "RGB"
        ):
            channel_names = ["Red", "Green", "Blue"]
            if image.shape[0] == 4:
                channel_names.append("Alpha")
    return MicroscopyImage(
        image=image,
        channel_names=channel_names,
        pixel_size_um=pixel_size,
        source_path=source,
    )


def load_ome_tiff(path: str | Path) -> MicroscopyImage:
    """Load TIFF or BMP microscopy data through the backward-compatible API.

    The historical name is retained so existing pipeline components and external
    callers continue working while BMP inputs use the same validated data model.
    """
    return load_microscopy_image(path)


def get_channel(microscopy_image: MicroscopyImage, channel_name: str) -> np.ndarray:
    """Select one microscopy channel without fuzzy or positional guessing.

    Exact metadata spelling is tried first, followed by a case-insensitive exact
    match. Missing, duplicated, or ambiguous names produce informative errors.
    """
    if not channel_name or not channel_name.strip():
        raise ValueError("channel_name must be a non-empty explicit channel name")
    exact = [i for i, name in enumerate(microscopy_image.channel_names) if name == channel_name]
    if len(exact) == 1:
        return microscopy_image.image[exact[0]]
    if len(exact) > 1:
        raise ValueError(f"Channel name {channel_name!r} is duplicated in the OME metadata.")
    folded = [
        i
        for i, name in enumerate(microscopy_image.channel_names)
        if name.casefold() == channel_name.casefold()
    ]
    if len(folded) == 1:
        return microscopy_image.image[folded[0]]
    if len(folded) > 1:
        raise ValueError(f"Case-insensitive channel name {channel_name!r} is ambiguous.")
    available = [name for name in microscopy_image.channel_names if name]
    raise KeyError(f"Channel {channel_name!r} was not found. Available named channels: {available}")
