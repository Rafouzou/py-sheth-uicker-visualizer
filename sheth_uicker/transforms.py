"""Core homogeneous-transform helpers (Steps 1–3 of the calculation pipeline)."""

import numpy as np


def rpy_to_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """Convert roll/pitch/yaw angles to a 3×3 rotation matrix.

    Convention
    ----------
    Intrinsic ZYX (equivalent to extrinsic XYZ):

      R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

    Rotations are applied in this order when composing:
    1. Roll  – rotation about the X-axis
    2. Pitch – rotation about the (new) Y-axis
    3. Yaw   – rotation about the (new) Z-axis

    All angles must be given in **radians**.

    Parameters
    ----------
    roll  : float  – rotation about X (radians)
    pitch : float  – rotation about Y (radians)
    yaw   : float  – rotation about Z (radians)

    Returns
    -------
    R : ndarray (3, 3)  rotation matrix
    """
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)

    Rx = np.array([[1,  0,   0 ],
                   [0,  cr, -sr],
                   [0,  sr,  cr]], dtype=float)

    Ry = np.array([[ cp, 0, sp],
                   [ 0,  1, 0 ],
                   [-sp, 0, cp]], dtype=float)

    Rz = np.array([[cy, -sy, 0],
                   [sy,  cy, 0],
                   [0,   0,  1]], dtype=float)

    return Rz @ Ry @ Rx


def build_homogeneous(R: np.ndarray, p: np.ndarray) -> np.ndarray:
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
