"""Tests for sheth_uicker/validation.py."""

import math

import numpy as np
import pytest

from sheth_uicker.transforms import (
    build_homogeneous,
    elementary_rotation,
    elementary_translation,
    relative_transform,
    rpy_to_matrix,
)
from sheth_uicker.decomposition import compute_sheth_uicker
from sheth_uicker.validation import reconstruct_transform, frobenius_error, decomposition_chain


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_T(pos, rpy=(0, 0, 0)):
    R = rpy_to_matrix(*rpy)
    return build_homogeneous(R, pos)


# ── reconstruct_transform ──────────────────────────────────────────────────────

class TestReconstructTransform:
    def test_all_zero_gives_identity(self):
        T = reconstruct_transform(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(T, np.eye(4), atol=1e-12)

    def test_shape_is_4x4(self):
        T = reconstruct_transform(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)
        assert T.shape == (4, 4)

    def test_bottom_row(self):
        T = reconstruct_transform(0.5, 1.0, 0.7, 2.0, 0.3, 0.5)
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])

    def test_result_is_rigid_body(self):
        """Upper-left 3×3 block must be a rotation matrix."""
        T = reconstruct_transform(0.5, 1.0, 0.7, 2.0, 0.3, 0.5)
        R = T[:3, :3]
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_pure_z_translation(self):
        """A1=A2=A3=L2=L3=0, L1=d → translation by d along Z."""
        d = 3.5
        T = reconstruct_transform(0.0, d, 0.0, 0.0, 0.0, 0.0)
        np.testing.assert_allclose(T[:3, 3], [0.0, 0.0, d], atol=1e-12)
        np.testing.assert_allclose(T[:3, :3], np.eye(3), atol=1e-12)

    def test_known_composition(self):
        """Explicit manual composition must match reconstruct_transform."""
        A1, L1, A2, L2, A3, L3 = 0.5, 1.0, 0.7, 2.0, 0.3, 0.5
        T_manual = (
            elementary_rotation("z", A1)
            @ elementary_translation("z", L1)
            @ elementary_rotation("x", A2)
            @ elementary_translation("x", L2)
            @ elementary_rotation("z", A3)
            @ elementary_translation("z", L3)
        )
        T_func = reconstruct_transform(A1, L1, A2, L2, A3, L3)
        np.testing.assert_allclose(T_func, T_manual, atol=1e-12)


# ── frobenius_error ────────────────────────────────────────────────────────────

class TestFrobeniusError:
    def test_identical_matrices_give_zero(self):
        T = reconstruct_transform(0.5, 1.0, 0.7, 2.0, 0.3, 0.5)
        assert frobenius_error(T, T) == pytest.approx(0.0, abs=1e-15)

    def test_identity_vs_shift(self):
        T1 = np.eye(4)
        T2 = np.eye(4)
        T2[0, 3] = 1.0
        # ||T2 - T1||_F = 1.0 (single entry differs by 1)
        assert frobenius_error(T1, T2) == pytest.approx(1.0, abs=1e-12)

    def test_symmetry(self):
        T1 = reconstruct_transform(0.5, 1.0, 0.7, 2.0, 0.3, 0.5)
        T2 = reconstruct_transform(1.0, 0.5, 0.3, 1.0, 0.7, 0.2)
        assert frobenius_error(T1, T2) == pytest.approx(frobenius_error(T2, T1), abs=1e-14)

    def test_returns_float(self):
        T = np.eye(4)
        result = frobenius_error(T, T)
        assert isinstance(result, float)


# ── decomposition_chain ────────────────────────────────────────────────────────

class TestDecompositionChain:
    def _params_from(self, T1, T2):
        return compute_sheth_uicker(T1, T2)

    def test_returns_seven_frames(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([1, 0, 0])
        p = self._params_from(T1, T2)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        assert len(chain) == 7

    def test_first_frame_is_T1(self):
        T1 = _make_T([2, 3, 4], (0.3, 0.5, 0.7))
        T2 = _make_T([-1, 0, 2], (1.0, 0.2, 1.5))
        p = self._params_from(T1, T2)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        np.testing.assert_allclose(chain[0], T1, atol=1e-12)

    def test_last_frame_equals_T2(self):
        """F6 must equal T2 up to floating-point reconstruction error."""
        T1 = _make_T([1, 2, 3], (0.3, 0.5, 0.7))
        T2 = _make_T([-1, 0, 2], (1.0, 0.2, 1.5))
        p = self._params_from(T1, T2)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        np.testing.assert_allclose(chain[6], T2, atol=1e-7)

    def test_all_frames_are_4x4(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([1, 1, 1], (0.5, 0.5, 0.5))
        p = self._params_from(T1, T2)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        for i, F in enumerate(chain):
            assert F.shape == (4, 4), f"Frame {i} has wrong shape {F.shape}"

    def test_all_frames_are_rigid_body(self):
        """Every frame must have an orthonormal rotation block and det = +1."""
        T1 = _make_T([1, 0, 0], (0.3, 0.0, 0.5))
        T2 = _make_T([0, 1, 0], (0.0, 0.7, 1.1))
        p = self._params_from(T1, T2)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        for i, F in enumerate(chain):
            R = F[:3, :3]
            np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-10,
                                       err_msg=f"Frame {i} rotation is not orthonormal")
            assert abs(np.linalg.det(R) - 1.0) < 1e-10, f"Frame {i} det(R) != 1"

    def test_identity_transform_chain_stays_at_T1(self):
        """When T1 == T2 every frame in the chain must equal T1."""
        T1 = _make_T([1, 2, 3], (0.3, 0.4, 0.5))
        p = self._params_from(T1, T1)
        chain = decomposition_chain(T1, p["A1"], p["L1"], p["A2"], p["L2"], p["A3"], p["L3"])
        for i, F in enumerate(chain):
            np.testing.assert_allclose(F, T1, atol=1e-7,
                                       err_msg=f"Frame {i} deviates from T1 for identity transform")
