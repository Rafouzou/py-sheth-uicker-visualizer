"""Tests for sheth_uicker/decomposition.py."""

import math

import numpy as np
import pytest

from sheth_uicker.transforms import (
    build_homogeneous,
    elementary_rotation,
    elementary_translation,
    relative_transform,
)
from sheth_uicker.decomposition import (
    decompose_zxz,
    rotation_zxz,
    solve_translations,
    canonicalize_parameters,
    compute_sheth_uicker,
)
from sheth_uicker.validation import reconstruct_transform, frobenius_error


# ── helpers ────────────────────────────────────────────────────────────────────

def _make_T(pos, rpy=(0, 0, 0)):
    """Build a homogeneous transform from a position and RPY angles."""
    from sheth_uicker.transforms import rpy_to_matrix
    R = rpy_to_matrix(*rpy)
    return build_homogeneous(R, pos)


def _assert_params_reconstruct(params, T1, T2, atol=1e-8):
    """Assert that params reconstruct T1⁻¹·T2 within tolerance."""
    T_rel_ref = relative_transform(T1, T2)
    T_rel_rec = reconstruct_transform(
        params["A1"], params["L1"],
        params["A2"], params["L2"],
        params["A3"], params["L3"],
    )
    err = frobenius_error(T_rel_ref, T_rel_rec)
    assert err < atol, f"Reconstruction error too large: {err}"


# ── decompose_zxz ──────────────────────────────────────────────────────────────

class TestDecomposeZxz:
    def test_identity_gives_zero_angles(self):
        A1, A2, A3 = decompose_zxz(np.eye(3))
        # When A2=0 (degenerate), convention sets A1=0 and A3 encodes the rotation.
        # For R=I that gives all zeros.
        R_back = rotation_zxz(A1, A2, A3)
        np.testing.assert_allclose(R_back, np.eye(3), atol=1e-10)

    def test_pure_z_rotation(self):
        """Rz(θ) → A1=0, A2=0, A3=θ (degenerate case)."""
        theta = 1.2
        R = elementary_rotation("z", theta)[:3, :3]
        A1, A2, A3 = decompose_zxz(R)
        R_back = rotation_zxz(A1, A2, A3)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_pure_x_rotation(self):
        """Rx(θ) → A1=0, A2=θ, A3=0."""
        theta = 0.8
        R = elementary_rotation("x", theta)[:3, :3]
        A1, A2, A3 = decompose_zxz(R)
        R_back = rotation_zxz(A1, A2, A3)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_general_rotation_roundtrip(self):
        """Random rotation: decompose then recompose should recover R."""
        from sheth_uicker.transforms import rpy_to_matrix
        R = rpy_to_matrix(0.4, 0.7, 1.1)
        A1, A2, A3 = decompose_zxz(R)
        R_back = rotation_zxz(A1, A2, A3)
        np.testing.assert_allclose(R_back, R, atol=1e-10)

    def test_a2_in_range_0_pi(self):
        """A2 must be in [0, π] for any rotation."""
        from sheth_uicker.transforms import rpy_to_matrix
        for angles in [(0.3, 0.5, 0.7), (1.2, 2.1, 3.0), (0, 0, 0)]:
            R = rpy_to_matrix(*angles)
            _, A2, _ = decompose_zxz(R)
            assert 0.0 <= A2 <= math.pi + 1e-12

    def test_rotation_zxz_is_so3(self):
        """rotation_zxz result must be a rotation matrix."""
        R = rotation_zxz(0.5, 1.0, 1.5)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12


# ── solve_translations ─────────────────────────────────────────────────────────

class TestSolveTranslations:
    def _roundtrip(self, A1, A2, A3, L1, L2, L3):
        """Build T from parameters, then solve_translations and compare.

        For non-degenerate cases (A2 ≠ 0) the result should match exactly.
        Degenerate cases (A2 = 0) are tested via compute_sheth_uicker instead.
        """
        T = reconstruct_transform(A1, L1, A2, L2, A3, L3)
        R_rel = T[:3, :3]
        p_rel = T[:3, 3]
        L1s, L2s, L3s = solve_translations(R_rel, p_rel, A1, A2, A3)
        np.testing.assert_allclose([L1s, L2s, L3s], [L1, L2, L3], atol=1e-10)

    def test_pure_translation_along_z_nondegenerate(self):
        """L1 translation with a non-zero A2 so the system is full-rank."""
        self._roundtrip(0.3, 0.7, 0.5, 1.5, 0.0, 0.0)

    def test_pure_translation_along_x(self):
        self._roundtrip(0.0, 0.0, 0.0, 0.0, 2.3, 0.0)

    def test_general_case(self):
        self._roundtrip(0.5, 1.0, 2.0, 0.5, 1.0, 0.3)

    def test_zero_translations(self):
        self._roundtrip(0.3, 0.0, 0.7, 0.0, 0.0, 0.0)


# ── canonicalize_parameters ───────────────────────────────────────────────────

class TestCanonicalizeParameters:
    def _check_bijection(self, A1, A2, A3, L1, L2, L3):
        """After canonicalization the transform must be unchanged and angle constraints satisfied."""
        T_orig = reconstruct_transform(A1, L1, A2, L2, A3, L3)
        cA1, cA2, cA3, cL1, cL2, cL3 = canonicalize_parameters(A1, A2, A3, L1, L2, L3)

        # Angle range constraints
        assert 0.0 <= cA1 < 2 * math.pi + 1e-10, f"A1={cA1} out of [0, 2π)"
        assert abs(cA2) < math.pi + 1e-10,        f"A2={cA2} out of (−π, π)"
        assert 0.0 <= cA3 < 2 * math.pi + 1e-10, f"A3={cA3} out of [0, 2π)"
        # L2 ≥ 0 constraint
        assert cL2 >= -1e-10, f"L2={cL2} must be non-negative"

        T_canon = reconstruct_transform(cA1, cL1, cA2, cL2, cA3, cL3)
        np.testing.assert_allclose(T_orig, T_canon, atol=1e-8,
                                   err_msg="Canonicalization changed the transform")

    def test_already_canonical(self):
        self._check_bijection(0.5, 1.0, 0.7, 1.0, 1.0, 0.5)

    def test_negative_a1(self):
        self._check_bijection(-0.5, 1.0, 0.7, 1.0, 1.0, 0.5)

    def test_a1_greater_than_2pi(self):
        self._check_bijection(7.0, 1.0, 0.7, 1.0, 1.0, 0.5)

    def test_negative_l2_flipped_to_positive(self):
        """Canonicalization must flip L2 to positive (and negate A2) when L2<0."""
        cA1, cA2, cA3, cL1, cL2, cL3 = canonicalize_parameters(0.3, 1.0, 0.5, 1.0, -2.0, 0.5)
        assert cL2 >= -1e-10, f"L2={cL2} must be non-negative after canonicalization"
        self._check_bijection(0.3, 1.0, 0.5, 1.0, -2.0, 0.5)

    def test_a2_zero_l3_absorbed(self):
        """When A2 ≈ 0, L3 must be absorbed into L1 and set to zero."""
        _, cA2, _, cL1, _, cL3 = canonicalize_parameters(0.0, 0.0, 0.0, 1.0, 0.0, 0.5)
        assert abs(cA2) < 1e-10
        assert abs(cL3) < 1e-10
        assert abs(cL1 - 1.5) < 1e-10  # L1 absorbed L3

    def test_a2_zero_l2_zero_a3_absorbed_into_a1(self):
        """When A2=0 and L2=0, A3 must be merged into A1 and zeroed."""
        cA1, cA2, cA3, _, cL2, _ = canonicalize_parameters(0.0, 0.0, 1.2, 1.0, 0.0, 0.0)
        assert abs(cA2) < 1e-10
        assert abs(cL2) < 1e-10
        assert abs(cA3) < 1e-10
        assert abs(cA1 - 1.2) < 1e-10  # A3 merged into A1


# ── compute_sheth_uicker ───────────────────────────────────────────────────────

class TestComputeShethUicker:
    def test_identical_frames_give_zero_params(self):
        T = _make_T([1, 2, 3], (0.3, 0.4, 0.5))
        params = compute_sheth_uicker(T, T)
        for k in ("L1", "L2", "L3"):
            assert abs(params[k]) < 1e-8, f"{k} should be ~0 for identical frames"
        for k in ("A1", "A2", "A3"):
            # A1 and A3 may both be zero or wrap to zero
            assert abs(params[k]) < 1e-8 or abs(params[k] - 2 * math.pi) < 1e-8

    def test_pure_translation(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([3, 0, 0])   # pure X translation expressed in world
        params = compute_sheth_uicker(T1, T2)
        _assert_params_reconstruct(params, T1, T2)

    def test_pure_z_translation(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([0, 0, 5])
        params = compute_sheth_uicker(T1, T2)
        _assert_params_reconstruct(params, T1, T2)

    def test_pure_rotation_about_z(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([0, 0, 0], (0, 0, math.pi / 4))
        params = compute_sheth_uicker(T1, T2)
        _assert_params_reconstruct(params, T1, T2)

    def test_general_transform(self):
        T1 = _make_T([1, 2, 3], (0.3, 0.5, 0.7))
        T2 = _make_T([-1, 0, 2], (1.0, 0.2, 1.5))
        params = compute_sheth_uicker(T1, T2)
        _assert_params_reconstruct(params, T1, T2)

    def test_canonical_constraints(self):
        T1 = _make_T([1, 0, 0])
        T2 = _make_T([0, 1, 0], (0.5, 0.7, 1.1))
        params = compute_sheth_uicker(T1, T2)
        assert 0.0 <= params["A1"] < 2 * math.pi + 1e-10
        assert abs(params["A2"]) < math.pi + 1e-10
        assert 0.0 <= params["A3"] < 2 * math.pi + 1e-10
        assert params["L2"] >= -1e-10

    def test_returns_all_six_keys(self):
        T1 = _make_T([0, 0, 0])
        T2 = _make_T([1, 0, 0])
        params = compute_sheth_uicker(T1, T2)
        assert set(params.keys()) == {"A1", "L1", "A2", "L2", "A3", "L3"}
