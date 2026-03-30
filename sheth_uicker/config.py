"""JSON configuration parsing for frame poses.

Supports loading source/destination frame poses from a JSON file.

Each frame can specify orientation either as:
  - ``rpy``: [roll, pitch, yaw] in radians
  - ``rotation_matrix``: 3×3 nested list

When **both** ``rpy`` and ``rotation_matrix`` are present for the same frame,
``rotation_matrix`` takes precedence and ``rpy`` is ignored.

Example JSON
------------
::

    {
        "source": {
            "position": [1.0, 0.0, 0.0],
            "rpy": [0.0, 0.0, 0.0]
        },
        "destination": {
            "position": [-1.0, 0.0, 0.0],
            "rotation_matrix": [[1,0,0],[0,1,0],[0,0,1]]
        }
    }
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np

from sheth_uicker.transforms import rpy_to_matrix


# ── public data types ──────────────────────────────────────────────────────────

class FramePose:
    """Parsed pose for a single coordinate frame."""

    def __init__(self, position: np.ndarray, rotation: np.ndarray) -> None:
        self.position: np.ndarray = position  # shape (3,)
        self.rotation: np.ndarray = rotation  # shape (3, 3)


class SceneConfig:
    """Parsed configuration for the full scene (source + destination frames)."""

    def __init__(self, source: FramePose, destination: FramePose) -> None:
        self.source = source
        self.destination = destination


# ── internal helpers ───────────────────────────────────────────────────────────

def _parse_position(raw: object, frame_name: str) -> np.ndarray:
    """Validate and return a (3,) position array."""
    try:
        arr = np.asarray(raw, dtype=float).flatten()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{frame_name}.position' must be a list of 3 numbers."
        ) from exc
    if arr.shape != (3,):
        raise ValueError(
            f"'{frame_name}.position' must have exactly 3 elements, got {arr.shape}."
        )
    return arr


def _parse_rotation(raw_frame: dict, frame_name: str) -> np.ndarray:
    """Return a (3, 3) rotation matrix from a frame dict.

    Precedence: ``rotation_matrix`` wins over ``rpy`` when both are present.
    """
    has_rm = "rotation_matrix" in raw_frame
    has_rpy = "rpy" in raw_frame

    if not has_rm and not has_rpy:
        # Neither key — default to identity rotation.
        return np.eye(3)

    if has_rm:
        try:
            R = np.asarray(raw_frame["rotation_matrix"], dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"'{frame_name}.rotation_matrix' must be a 3×3 nested list of numbers."
            ) from exc
        if R.shape != (3, 3):
            raise ValueError(
                f"'{frame_name}.rotation_matrix' must be 3×3, got shape {R.shape}."
            )
        return R

    # Use rpy
    try:
        rpy = np.asarray(raw_frame["rpy"], dtype=float).flatten()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"'{frame_name}.rpy' must be a list of 3 numbers (radians)."
        ) from exc
    if rpy.shape != (3,):
        raise ValueError(
            f"'{frame_name}.rpy' must have exactly 3 elements, got {rpy.shape}."
        )
    return rpy_to_matrix(rpy[0], rpy[1], rpy[2])


def _parse_frame(raw_frame: object, frame_name: str) -> FramePose:
    """Parse a single frame object from the JSON config."""
    if not isinstance(raw_frame, dict):
        raise ValueError(
            f"'{frame_name}' must be a JSON object (dict), got {type(raw_frame).__name__}."
        )

    position = _parse_position(raw_frame.get("position", [0.0, 0.0, 0.0]), frame_name)
    rotation = _parse_rotation(raw_frame, frame_name)
    return FramePose(position=position, rotation=rotation)


# ── public API ─────────────────────────────────────────────────────────────────

def load_config(path: str | Path) -> SceneConfig:
    """Load and validate a JSON configuration file.

    Parameters
    ----------
    path : str or Path
        Path to the JSON file.

    Returns
    -------
    SceneConfig
        Parsed scene configuration with source and destination :class:`FramePose`.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the JSON content fails validation.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        try:
            data = json.load(fh)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid JSON in config file '{path}': {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")

    source = _parse_frame(data.get("source", {}), "source")
    destination = _parse_frame(data.get("destination", {}), "destination")

    return SceneConfig(source=source, destination=destination)
