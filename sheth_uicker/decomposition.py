"""Sheth-Uicker decomposition (Steps 4–6 of the calculation pipeline).

The rigid-body transformation T_rel is factored as:

    T_rel = Rz(A1) · Tz(L1) · Rx(A2) · Tx(L2) · Rz(A3) · Tz(L3)

Functions
---------
decompose_zxz         — ZXZ Euler-angle decomposition of a rotation matrix
rotation_zxz          — forward ZXZ composition (verification helper)
solve_translations    — extract L1, L2, L3 given the angles
canonicalize_parameters — enforce unique / canonical representation
compute_sheth_uicker  — full pipeline: T1, T2 → six parameters
"""

from __future__ import annotations

import numpy as np

from sheth_uicker.transforms import (
    relative_transform,
    extract_rotation,
    extract_translation,
    elementary_rotation,
)

_TWO_PI = 2.0 * np.pi
_ANGLE_TOL = 1e-10   # tolerance for treating an angle as zero (radians)
_DET_TOL = 1e-12     # tolerance for treating a determinant as zero


# ── ZXZ Euler decomposition ────────────────────────────────────────────────────

def decompose_zxz(R: np.ndarray) -> tuple[float, float, float]:
    """Decompose a 3×3 rotation matrix into ZXZ Euler angles (A1, A2, A3).

    The decomposition satisfies:

        R = Rz(A1) · Rx(A2) · Rz(A3)

    Parameters
    ----------
    R : ndarray (3, 3)

    Returns
    -------
    A1, A2, A3 : float
        Angles in radians.  Returned in the **un-canonicalized** range
        (A1, A3 ∈ (−π, π]; A2 ∈ [0, π]).
        Call :func:`canonicalize_parameters` to get the canonical form.
    """
    R = np.asarray(R, dtype=float)

    # A2 ∈ [0, π]
    A2 = np.arctan2(np.sqrt(R[2, 0] ** 2 + R[2, 1] ** 2), R[2, 2])

    if abs(np.sin(A2)) > 1e-10:  # non-degenerate
        # From R = Rz(A1)·Rx(A2)·Rz(A3):
        #   R[0,2] =  sin(A1)*sin(A2),  R[1,2] = -cos(A1)*sin(A2)
        #   R[2,0] =  sin(A2)*sin(A3),  R[2,1] =  sin(A2)*cos(A3)
        A1 = np.arctan2( R[0, 2] / np.sin(A2), -R[1, 2] / np.sin(A2))
        A3 = np.arctan2( R[2, 0] / np.sin(A2),  R[2, 1] / np.sin(A2))
    else:
        # Gimbal lock: only A1 ± A3 is determined; convention: A1 = 0.
        A1 = 0.0
        if R[2, 2] > 0.0:   # A2 ≈ 0  →  R ≈ Rz(A1+A3)
            # R[1,0]=sin(A1+A3), R[0,0]=cos(A1+A3)
            A3 = np.arctan2(R[1, 0], R[0, 0])
        else:                # A2 ≈ π  →  only A1−A3 is encoded
            # R[1,0]=sin(A1−A3), R[0,0]=cos(A1−A3); set A3 = A1 − (A1−A3)
            A3 = np.arctan2(-R[1, 0], R[0, 0])

    return float(A1), float(A2), float(A3)


def rotation_zxz(A1: float, A2: float, A3: float) -> np.ndarray:
    """Build a 3×3 rotation matrix from ZXZ Euler angles.

        R = Rz(A1) · Rx(A2) · Rz(A3)

    Parameters
    ----------
    A1, A2, A3 : float  — angles in radians

    Returns
    -------
    R : ndarray (3, 3)
    """
    Rz_A1 = elementary_rotation("z", A1)[:3, :3]
    Rx_A2 = elementary_rotation("x", A2)[:3, :3]
    Rz_A3 = elementary_rotation("z", A3)[:3, :3]
    return Rz_A1 @ Rx_A2 @ Rz_A3


# ── Translation solver ─────────────────────────────────────────────────────────

def solve_translations(
    R_rel: np.ndarray,
    p_rel: np.ndarray,
    A1: float,
    A2: float,
    A3: float,
) -> tuple[float, float, float]:
    """Solve for the translation parameters L1, L2, L3.

    Given the ZXZ Euler angles A1, A2, A3 and the relative position p_rel,
    the three translations satisfy:

        p_rel = Rz(A1)·(L1·ẑ + Rx(A2)·(L2·x̂ + Rz(A3)·L3·ẑ))

    Solved by back-substitution (propagating from right to left):

        p1 = Rz(A1)ᵀ · p_rel
        L3 = (Rz(A3)ᵀ · (Rx(A2)ᵀ · (p1 − L1·ẑ) − L2·x̂))[2]

    The linear system for (L1, L2, L3) is assembled directly and solved
    with ``numpy.linalg.solve``, which is numerically robust.

    Parameters
    ----------
    R_rel : ndarray (3, 3) — relative rotation matrix (unused; kept for API symmetry)
    p_rel : ndarray (3,)   — relative position vector
    A1, A2, A3 : float     — ZXZ Euler angles (radians)

    Returns
    -------
    L1, L2, L3 : float
    """
    p_rel = np.asarray(p_rel, dtype=float)

    Rz_A1 = elementary_rotation("z", A1)[:3, :3]
    Rx_A2 = elementary_rotation("x", A2)[:3, :3]
    Rz_A3 = elementary_rotation("z", A3)[:3, :3]

    # Unit vectors of each local axis after the respective rotation
    # expressed in the world frame.
    z_hat = np.array([0.0, 0.0, 1.0])
    x_hat = np.array([1.0, 0.0, 0.0])

    # Column vectors for the system M·[L1, L2, L3]ᵀ = p_rel
    col_L1 = Rz_A1 @ z_hat                    # direction of L1 in world
    col_L2 = Rz_A1 @ (Rx_A2 @ x_hat)         # direction of L2 in world
    col_L3 = Rz_A1 @ (Rx_A2 @ (Rz_A3 @ z_hat))  # direction of L3 in world

    M = np.column_stack([col_L1, col_L2, col_L3])  # (3, 3)

    # Fall back to least-squares when the system is rank-deficient
    # (degenerate configurations such as A2 = 0 with L2 = 0).
    if abs(np.linalg.det(M)) > _DET_TOL:
        L1, L2, L3 = np.linalg.solve(M, p_rel)
    else:
        L1, L2, L3 = np.linalg.lstsq(M, p_rel, rcond=None)[0]

    return float(L1), float(L2), float(L3)


# ── Canonicalization ───────────────────────────────────────────────────────────

def canonicalize_parameters(
    A1: float,
    A2: float,
    A3: float,
    L1: float,
    L2: float,
    L3: float,
) -> tuple[float, float, float, float, float, float]:
    """Enforce the bijection constraints to produce a canonical parameter set.

    Constraints applied (in order):

    1. A2 normalised to [0, 2π) then folded to [0, π)
    2. A1, A3 wrapped to [0, 2π)
    3. (A2 = 0) ⇒ (L3 = 0)          — L3 is absorbed into L1
    4. (L2 = 0 ∧ A2 = 0) ⇒ (A3 = 0) — A3 is absorbed into A1

    Note on L2: when A2 ∈ (0, π) the linear system for the translations has a
    unique solution which may yield L2 < 0.  No additional sign constraint is
    enforced here; the sign of L2 is part of the unique canonical representation
    (it encodes the directed offset along the common perpendicular between the
    two Z-axes).

    Parameters
    ----------
    A1, A2, A3 : float   — ZXZ angles (radians, any range)
    L1, L2, L3 : float   — translation parameters

    Returns
    -------
    (A1, A2, A3, L1, L2, L3) : tuple[float, float, float, float, float, float]
        Canonical parameters.
    """
    # ── enforce A2 ∈ [0, π) ───────────────────────────────────────────────────
    # Normalise A2 to [0, 2π) first, then fold into [0, π).
    # Equivalence: Rz(A1)·Rx(A2)·Rz(A3) = Rz(A1+π)·Rx(2π−A2)·Rz(A3+π)
    # Note: this rotation equivalence only applies to the rotation part; when
    # A2 is already in [0, π) after the atan2 decomposition (as enforced by
    # decompose_zxz), no fold is needed.  The modulo handles edge-cases where
    # callers pass angles outside [0, 2π).
    A2 = A2 % _TWO_PI
    if A2 >= np.pi:
        A2 = _TWO_PI - A2
        A1 = A1 + np.pi
        A3 = A3 + np.pi

    # ── wrap A1, A3 to [0, 2π) ────────────────────────────────────────────────
    A1 = A1 % _TWO_PI
    A3 = A3 % _TWO_PI

    # ── gimbal-lock special cases ─────────────────────────────────────────────

    if abs(A2) < _ANGLE_TOL:  # A2 ≈ 0
        A2 = 0.0
        # When A2=0, Tz(L3) along the final Z″ is indistinguishable from an
        # extra Tz contribution along the Z axis rotated by A1. Absorb L3 → L1.
        L1 = L1 + L3
        L3 = 0.0
        # When L2=0 as well, only A1+A3 is distinguishable (gimbal lock).
        # Convention: encode the combined rotation in A1 and set A3=0.
        if abs(L2) < _ANGLE_TOL:
            A1 = (A1 + A3) % _TWO_PI
            A3 = 0.0

    return float(A1), float(A2), float(A3), float(L1), float(L2), float(L3)


# ── Full pipeline ──────────────────────────────────────────────────────────────

def compute_sheth_uicker(
    T1: np.ndarray,
    T2: np.ndarray,
) -> dict[str, float]:
    """Compute the Sheth-Uicker parameters for the transformation T1 → T2.

    The six parameters (A1, L1, A2, L2, A3, L3) satisfy:

        T1⁻¹ · T2 = Rz(A1) · Tz(L1) · Rx(A2) · Tx(L2) · Rz(A3) · Tz(L3)

    Parameters
    ----------
    T1, T2 : ndarray (4, 4)

    Returns
    -------
    dict with keys 'A1', 'L1', 'A2', 'L2', 'A3', 'L3'
    """
    T_rel = relative_transform(T1, T2)
    R_rel = extract_rotation(T_rel)
    p_rel = extract_translation(T_rel)

    A1, A2, A3 = decompose_zxz(R_rel)
    L1, L2, L3 = solve_translations(R_rel, p_rel, A1, A2, A3)
    A1, A2, A3, L1, L2, L3 = canonicalize_parameters(A1, A2, A3, L1, L2, L3)

    return {"A1": A1, "L1": L1, "A2": A2, "L2": L2, "A3": A3, "L3": L3}
