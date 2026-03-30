"""Tests for sheth_uicker/transforms.py."""

import math

import numpy as np
import pytest

from sheth_uicker.transforms import rpy_to_matrix, build_homogeneous


class TestRpyToMatrix:
    """Tests for rpy_to_matrix()."""

    def test_zero_angles_gives_identity(self):
        R = rpy_to_matrix(0.0, 0.0, 0.0)
        np.testing.assert_allclose(R, np.eye(3), atol=1e-12)

    def test_90_roll_about_x(self):
        """90° roll only → Rx(90°)."""
        R = rpy_to_matrix(math.pi / 2, 0.0, 0.0)
        expected = np.array([[1, 0,  0],
                              [0, 0, -1],
                              [0, 1,  0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90_pitch_about_y(self):
        """90° pitch only → Ry(90°)."""
        R = rpy_to_matrix(0.0, math.pi / 2, 0.0)
        expected = np.array([[ 0, 0, 1],
                              [ 0, 1, 0],
                              [-1, 0, 0]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_90_yaw_about_z(self):
        """90° yaw only → Rz(90°)."""
        R = rpy_to_matrix(0.0, 0.0, math.pi / 2)
        expected = np.array([[0, -1, 0],
                              [1,  0, 0],
                              [0,  0, 1]], dtype=float)
        np.testing.assert_allclose(R, expected, atol=1e-12)

    def test_result_is_rotation_matrix(self):
        """Result must be orthogonal with det = +1 for arbitrary angles."""
        R = rpy_to_matrix(0.3, 0.7, 1.2)
        np.testing.assert_allclose(R @ R.T, np.eye(3), atol=1e-12)
        assert abs(np.linalg.det(R) - 1.0) < 1e-12

    def test_shape(self):
        R = rpy_to_matrix(0.1, 0.2, 0.3)
        assert R.shape == (3, 3)

    def test_order_zyx(self):
        """R(roll, pitch, yaw) == Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
        roll, pitch, yaw = 0.4, 0.5, 0.6
        r = rpy_to_matrix(roll, pitch, yaw)

        cr, sr = math.cos(roll), math.sin(roll)
        cp, sp = math.cos(pitch), math.sin(pitch)
        cy, sy = math.cos(yaw), math.sin(yaw)

        Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
        Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
        Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        expected = Rz @ Ry @ Rx

        np.testing.assert_allclose(r, expected, atol=1e-12)


class TestBuildHomogeneous:
    """Tests for build_homogeneous() (existing function, regression coverage)."""

    def test_identity_rotation_and_zero_pos(self):
        T = build_homogeneous(np.eye(3), [0, 0, 0])
        np.testing.assert_allclose(T, np.eye(4))

    def test_position_stored_correctly(self):
        T = build_homogeneous(np.eye(3), [1.0, 2.0, 3.0])
        np.testing.assert_allclose(T[:3, 3], [1.0, 2.0, 3.0])

    def test_rotation_stored_correctly(self):
        R = rpy_to_matrix(0.0, 0.0, math.pi / 2)
        T = build_homogeneous(R, [0, 0, 0])
        np.testing.assert_allclose(T[:3, :3], R, atol=1e-12)

    def test_bottom_row(self):
        T = build_homogeneous(np.eye(3), [5, 6, 7])
        np.testing.assert_allclose(T[3, :], [0, 0, 0, 1])
