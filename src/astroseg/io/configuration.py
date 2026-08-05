"""YAML configuration loading with portable environment-backed paths."""

import os
import re
from collections.abc import Collection, Mapping
from pathlib import Path
from typing import Any

import yaml


_ENVIRONMENT_REFERENCE = re.compile(r"\$(?:\{(?P<braced>[A-Za-z_][A-Za-z0-9_]*)\}|(?P<plain>[A-Za-z_][A-Za-z0-9_]*))")


def _expand_environment(value: Any) -> Any:
    """Recursively expand environment variables and user-home path markers.

    Undefined variables fail explicitly instead of surviving inside a path and
    producing a misleading file-not-found error later on a compute node.
    """
    if isinstance(value, str):
        missing = {
            match.group("braced") or match.group("plain")
            for match in _ENVIRONMENT_REFERENCE.finditer(value)
            if (match.group("braced") or match.group("plain")) not in os.environ
        }
        if missing:
            raise ValueError(f"Undefined configuration environment variables: {sorted(missing)}")
        return os.path.expanduser(os.path.expandvars(value))
    if isinstance(value, list):
        return [_expand_environment(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_environment(item) for item in value)
    if isinstance(value, Mapping):
        return {key: _expand_environment(item) for key, item in value.items()}
    return value


def load_yaml_configuration(
    path: str | Path,
    required_sections: Collection[str] = (),
) -> dict[str, Any]:
    """Load a YAML mapping, expand portable paths, and validate named sections.

    Cluster configurations can use variables such as ``${ASTROSEG_DATA_ROOT}``
    without hard-coding one account path. Required sections must be mappings.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"Configuration does not exist: {source}")
    with source.open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Configuration must be a YAML mapping")
    expanded = _expand_environment(value)
    for section in required_sections:
        if not isinstance(expanded.get(section), dict):
            raise ValueError(f"Configuration is missing mapping section {section!r}")
    return expanded
