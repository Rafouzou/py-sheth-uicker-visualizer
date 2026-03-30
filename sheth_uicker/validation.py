"""Reconstruction and validation utilities (Step 7 of the calculation pipeline).

Functions
---------
reconstruct_transform  — build the 4×4 transform from the six parameters
frobenius_error        — measure distance between two matrices
decomposition_chain    — generate the seven intermediate frames F0 … F6
"""

from __future__ import annotations

import numpy as np

from sheth_uicker.transforms import (
    elementary_rotation,
    elementary_translation,
)


def reconstruct_transform(
    A1: float,
    L1: float,
    A2: float,
    L2: float,
    A3: float,
    L3: float,
) -> np.ndarray:
    """Reconstruct the full 4×4 relative transformation from Sheth-Uicker parameters.

        T = Rz(A1) · Tz(L1) · Rx(A2) · Tx(L2) · Rz(A3) · Tz(L3)

    Parameters
    ----------
    A1, L1, A2, L2, A3, L3 : float

    Returns
    -------
    T : ndarray (4, 4)
    """
    return (
        elementary_rotation("z", A1)
        @ elementary_translation("z", L1)
        @ elementary_rotation("x", A2)
        @ elementary_translation("x", L2)
        @ elementary_rotation("z", A3)
        @ elementary_translation("z", L3)
    )


def frobenius_error(T_a: np.ndarray, T_b: np.ndarray) -> float:
    """Return the Frobenius norm of the difference between two matrices.

    Parameters
    ----------
    T_a, T_b : ndarray (4, 4)

    Returns
    -------
    error : float
    """
    return float(np.linalg.norm(T_a - T_b, "fro"))


def decomposition_chain(
    T1: np.ndarray,
    A1: float,
    L1: float,
    A2: float,
    L2: float,
    A3: float,
    L3: float,
) -> list[np.ndarray]:
    """Generate the seven intermediate frames along the Sheth-Uicker decomposition path.

    Starting from *T1* each of the six elementary transformations is applied in
    the **local** (right-multiplication) sense:

        F0 = T1
        F1 = F0 · Rz(A1)
        F2 = F1 · Tz(L1)
        F3 = F2 · Rx(A2)
        F4 = F3 · Tx(L2)
        F5 = F4 · Rz(A3)
        F6 = F5 · Tz(L3)   ← should equal T2 up to floating-point error

    Parameters
    ----------
    T1 : ndarray (4, 4) — source frame
    A1, L1, A2, L2, A3, L3 : float — Sheth-Uicker parameters

    Returns
    -------
    chain : list of 7 ndarray (4, 4) — frames F0 … F6
    """
    F0 = np.asarray(T1, dtype=float).copy()
    F1 = F0 @ elementary_rotation("z", A1)
    F2 = F1 @ elementary_translation("z", L1)
    F3 = F2 @ elementary_rotation("x", A2)
    F4 = F3 @ elementary_translation("x", L2)
    F5 = F4 @ elementary_rotation("z", A3)
    F6 = F5 @ elementary_translation("z", L3)
    return [F0, F1, F2, F3, F4, F5, F6]
