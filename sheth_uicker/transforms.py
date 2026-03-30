"""Core homogeneous-transform helpers (Steps 1–3 of the calculation pipeline)."""

import numpy as np


def build_homogeneous(R: np.ndarray, p) -> np.ndarray:
    """Assemble a 4×4 homogeneous transformation matrix from a rotation and a position.

    Parameters
    ----------
    R : array-like (3, 3)
        Rotation matrix whose columns are the X, Y, Z unit vectors of the frame
        expressed in the reference frame.
    p : array-like (3,)
        Origin of the frame in the reference frame.

    Returns
    -------
    T : ndarray (4, 4)
    """
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def identity_frame() -> np.ndarray:
    """Return a 4×4 identity transformation (origin at zero, axes aligned with world)."""
    return np.eye(4)
