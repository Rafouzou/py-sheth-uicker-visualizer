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


def invert_homogeneous(T: np.ndarray) -> np.ndarray:
    """Compute the inverse of a rigid-body (homogeneous) transformation matrix.

    Uses the closed-form inverse for SE(3):

        T⁻¹ = | Rᵀ  −Rᵀ·p |
              |  0     1   |

    Parameters
    ----------
    T : ndarray (4, 4)
        Rigid-body transformation matrix.

    Returns
    -------
    T_inv : ndarray (4, 4)
    """
    R = T[:3, :3]
    p = T[:3, 3]
    T_inv = np.eye(4)
    T_inv[:3, :3] = R.T
    T_inv[:3, 3] = -R.T @ p
    return T_inv


def relative_transform(T1: np.ndarray, T2: np.ndarray) -> np.ndarray:
    """Return the transformation that maps frame 1 into frame 2, expressed in frame 1.

        T_rel = T1⁻¹ · T2

    Parameters
    ----------
    T1, T2 : ndarray (4, 4)
        Source and destination homogeneous transforms.

    Returns
    -------
    T_rel : ndarray (4, 4)
    """
    return invert_homogeneous(T1) @ T2


def extract_rotation(T: np.ndarray) -> np.ndarray:
    """Return the 3×3 rotation block of a 4×4 homogeneous matrix."""
    return T[:3, :3].copy()


def extract_translation(T: np.ndarray) -> np.ndarray:
    """Return the 3-vector translation column of a 4×4 homogeneous matrix."""
    return T[:3, 3].copy()


def elementary_rotation(axis: str, angle: float) -> np.ndarray:
    """Build a 4×4 homogeneous matrix for a pure rotation about a world axis.

    Parameters
    ----------
    axis  : 'x', 'y', or 'z'
    angle : rotation angle in radians

    Returns
    -------
    T : ndarray (4, 4)
    """
    c, s = np.cos(angle), np.sin(angle)
    axis = axis.lower()
    if axis == "x":
        R = np.array([[1, 0,  0],
                      [0, c, -s],
                      [0, s,  c]], dtype=float)
    elif axis == "y":
        R = np.array([[ c, 0, s],
                      [ 0, 1, 0],
                      [-s, 0, c]], dtype=float)
    elif axis == "z":
        R = np.array([[c, -s, 0],
                      [s,  c, 0],
                      [0,  0, 1]], dtype=float)
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got '{axis}'")
    return build_homogeneous(R, [0.0, 0.0, 0.0])


def elementary_translation(axis: str, distance: float) -> np.ndarray:
    """Build a 4×4 homogeneous matrix for a pure translation along a world axis.

    Parameters
    ----------
    axis     : 'x', 'y', or 'z'
    distance : translation distance

    Returns
    -------
    T : ndarray (4, 4)
    """
    axis = axis.lower()
    if axis == "x":
        p = [distance, 0.0, 0.0]
    elif axis == "y":
        p = [0.0, distance, 0.0]
    elif axis == "z":
        p = [0.0, 0.0, distance]
    else:
        raise ValueError(f"axis must be 'x', 'y', or 'z'; got '{axis}'")
    return build_homogeneous(np.eye(3), p)
