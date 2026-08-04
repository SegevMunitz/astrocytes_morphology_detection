"""Automatic selection of model channels from microscopy images."""

from dataclasses import dataclass

import numpy as np

from astroseg.io.ome_tiff import MicroscopyImage, get_channel
from astroseg.preprocessing.normalize import percentile_normalize


@dataclass(frozen=True)
class ChannelSelection:
    """Resolved GFAP and DAPI names with auditable selection information.

    Explicit manifest values take priority. RGB composites without biological
    metadata use Blue for DAPI and the stronger of Red or Green for GFAP.
    """

    gfap_channel: str
    dapi_channel: str
    method: str
    red_score: float | None = None
    green_score: float | None = None


def _find_named_channel(microscopy_image: MicroscopyImage, aliases: tuple[str, ...]) -> str | None:
    """Return one channel matching ordered case-insensitive aliases.

    Alias order expresses preference, and duplicated matching names fail through
    the shared exact channel accessor rather than being chosen arbitrarily.
    """
    names_by_folded = {name.casefold(): name for name in microscopy_image.channel_names if name}
    for alias in aliases:
        name = names_by_folded.get(alias.casefold())
        if name is not None:
            get_channel(microscopy_image, name)
            return name
    return None


def _signal_score(channel: np.ndarray) -> float:
    """Measure robust high-intensity contrast for automatic RGB selection.

    The 99.8th percentile minus the median favors a sparse fluorescent signal and
    is insensitive to a small number of saturated pixels.
    """
    if channel.ndim != 2 or not np.issubdtype(channel.dtype, np.number):
        raise ValueError("Channel scoring requires a numeric 2D array")
    if not np.isfinite(channel).all():
        raise ValueError("Channel scoring does not accept non-finite values")
    median, upper = np.percentile(channel, (50.0, 99.8))
    return float(max(0.0, upper - median))


def select_model_channels(
    microscopy_image: MicroscopyImage,
    gfap_channel: str = "",
    dapi_channel: str = "",
) -> ChannelSelection:
    """Resolve GFAP and DAPI channels without per-image manual extraction.

    Manifest names are validated and retained when present. Named OME channels
    are detected next; RGB composites fall back to Blue plus dominant Red/Green.
    """
    explicit_gfap = gfap_channel.strip()
    explicit_dapi = dapi_channel.strip()
    if explicit_gfap:
        get_channel(microscopy_image, explicit_gfap)
    if explicit_dapi:
        get_channel(microscopy_image, explicit_dapi)
    if explicit_gfap and explicit_dapi:
        if explicit_gfap.casefold() == explicit_dapi.casefold():
            raise ValueError("GFAP and DAPI must refer to different channels")
        return ChannelSelection(explicit_gfap, explicit_dapi, "manifest")

    resolved_dapi = explicit_dapi or _find_named_channel(
        microscopy_image, ("DAPI", "Hoechst", "Blue")
    )
    resolved_gfap = explicit_gfap or _find_named_channel(microscopy_image, ("GFAP",))
    if resolved_gfap and resolved_dapi:
        if resolved_gfap.casefold() == resolved_dapi.casefold():
            raise ValueError("Automatically selected GFAP and DAPI channels are identical")
        return ChannelSelection(resolved_gfap, resolved_dapi, "named_metadata")

    names = {name.casefold(): name for name in microscopy_image.channel_names if name}
    if {"red", "green", "blue"}.issubset(names):
        resolved_dapi = resolved_dapi or names["blue"]
        red_name = names["red"]
        green_name = names["green"]
        red_score = _signal_score(get_channel(microscopy_image, red_name))
        green_score = _signal_score(get_channel(microscopy_image, green_name))
        if resolved_gfap is None:
            if max(red_score, green_score) <= 0:
                raise ValueError("Neither Red nor Green contains measurable fluorescence signal")
            resolved_gfap = green_name if green_score >= red_score else red_name
        if resolved_gfap.casefold() == resolved_dapi.casefold():
            raise ValueError("Automatically selected GFAP and DAPI channels are identical")
        return ChannelSelection(
            resolved_gfap,
            resolved_dapi,
            "rgb_signal",
            red_score=red_score,
            green_score=green_score,
        )

    raise ValueError(
        "Could not determine GFAP and DAPI channels automatically. "
        f"Available names: {microscopy_image.channel_names}"
    )


def prepare_dapi_for_detection(
    dapi: np.ndarray,
    gfap: np.ndarray,
    selection: ChannelSelection,
) -> tuple[np.ndarray, str]:
    """Suppress GFAP color mixing before detecting nuclei in RGB composites.

    Pink/red or cyan renderings can place the structural marker in the Blue sample
    as well as Red/Green. Subtracting normalized GFAP removes those processes;
    biologically named OME channels retain their original numeric values.
    """
    if selection.dapi_channel.casefold() != "blue" or selection.gfap_channel.casefold() not in {
        "red",
        "green",
    }:
        return np.asarray(dapi), "native_dapi"
    corrected = np.clip(
        percentile_normalize(dapi) - percentile_normalize(gfap),
        0.0,
        1.0,
    )
    return corrected.astype(np.float32, copy=False), "normalized_blue_minus_gfap"
