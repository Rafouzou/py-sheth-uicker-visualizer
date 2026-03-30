"""Tests for sheth_uicker/config.py and main.py CLI parsing."""

import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from sheth_uicker.config import load_config, FramePose, SceneConfig


# ── Helpers ────────────────────────────────────────────────────────────────────

def _write_config(tmp_path: Path, data: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ── load_config: rpy ───────────────────────────────────────────────────────────

class TestLoadConfigRpy:
    def test_zero_rpy_gives_identity_rotation(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
            "destination": {"position": [-1.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.rotation, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(cfg.destination.rotation, np.eye(3), atol=1e-12)

    def test_position_parsed(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [2.0, 3.0, 4.0], "rpy": [0.0, 0.0, 0.0]},
            "destination": {"position": [-2.0, -3.0, -4.0], "rpy": [0.0, 0.0, 0.0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.position, [2.0, 3.0, 4.0])
        np.testing.assert_allclose(cfg.destination.position, [-2.0, -3.0, -4.0])

    def test_90_yaw_rotation(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [0, 0, 0], "rpy": [0.0, 0.0, math.pi / 2]},
            "destination": {"position": [0, 0, 0], "rpy": [0.0, 0.0, 0.0]},
        })
        cfg = load_config(cfg_path)
        expected = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(cfg.source.rotation, expected, atol=1e-12)


# ── load_config: rotation_matrix ──────────────────────────────────────────────

class TestLoadConfigRotationMatrix:
    def test_identity_rotation_matrix(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {
                "position": [1.0, 0.0, 0.0],
                "rotation_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            },
            "destination": {"position": [-1.0, 0.0, 0.0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.rotation, np.eye(3), atol=1e-12)

    def test_custom_rotation_matrix(self, tmp_path):
        """A 90° yaw matrix supplied directly."""
        R = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [0, 0, 0], "rotation_matrix": R},
            "destination": {"position": [0, 0, 0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.rotation, np.array(R, dtype=float), atol=1e-12)


# ── load_config: precedence (rotation_matrix wins over rpy) ───────────────────

class TestLoadConfigPrecedence:
    def test_rotation_matrix_wins_over_rpy(self, tmp_path):
        """When both rpy and rotation_matrix are given, rotation_matrix must win."""
        identity_rm = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        cfg_path = _write_config(tmp_path, {
            "source": {
                "position": [0, 0, 0],
                "rpy": [math.pi / 2, 0.0, 0.0],  # non-identity rotation
                "rotation_matrix": identity_rm,    # identity wins
            },
            "destination": {"position": [0, 0, 0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.rotation, np.eye(3), atol=1e-12)


# ── load_config: missing keys default ─────────────────────────────────────────

class TestLoadConfigDefaults:
    def test_missing_source_defaults_to_origin(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "destination": {"position": [-1, 0, 0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.position, [0.0, 0.0, 0.0])
        np.testing.assert_allclose(cfg.source.rotation, np.eye(3), atol=1e-12)

    def test_missing_rotation_defaults_to_identity(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [1, 0, 0]},
            "destination": {"position": [-1, 0, 0]},
        })
        cfg = load_config(cfg_path)
        np.testing.assert_allclose(cfg.source.rotation, np.eye(3), atol=1e-12)


# ── load_config: validation errors ────────────────────────────────────────────

class TestLoadConfigValidation:
    def test_file_not_found(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_config(tmp_path / "nonexistent.json")

    def test_invalid_json(self, tmp_path):
        p = tmp_path / "bad.json"
        p.write_text("{not valid json", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid JSON"):
            load_config(p)

    def test_position_wrong_length(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [1.0, 0.0]},  # only 2 elements
        })
        with pytest.raises(ValueError, match="3 elements"):
            load_config(cfg_path)

    def test_rpy_wrong_length(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"rpy": [0.0, 0.0]},  # only 2 elements
        })
        with pytest.raises(ValueError, match="3 elements"):
            load_config(cfg_path)

    def test_rotation_matrix_wrong_shape(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"rotation_matrix": [[1, 0], [0, 1]]},  # 2×2
        })
        with pytest.raises(ValueError, match="3.3"):
            load_config(cfg_path)

    def test_top_level_not_dict(self, tmp_path):
        p = tmp_path / "list.json"
        p.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(ValueError, match="JSON object"):
            load_config(p)


# ── main.py CLI: precedence defaults < config < CLI ───────────────────────────

class TestMainCliPrecedence:
    """Test precedence: defaults → config → CLI flags."""

    def _parse_frames(self, argv, tmp_path=None):
        """Import and run main's parser, return built transforms."""
        # We test build logic, not rendering, so we mock render_scene.
        import importlib
        import unittest.mock as mock

        captured = {}

        with mock.patch("sheth_uicker.visualisation.render_scene") as mocked:
            mocked.side_effect = lambda T_src, T_dst, **kw: captured.update(
                {"T_source": T_src, "T_dest": T_dst}
            )
            import main as m
            importlib.reload(m)  # ensure clean state
            m.main(argv)

        return captured["T_source"], captured["T_dest"]

    def test_defaults_no_args(self, tmp_path):
        T_src, T_dst = self._parse_frames([])
        np.testing.assert_allclose(T_src[:3, 3], [1.0, 0.0, 0.0])
        np.testing.assert_allclose(T_dst[:3, 3], [-1.0, 0.0, 0.0])
        np.testing.assert_allclose(T_src[:3, :3], np.eye(3), atol=1e-12)
        np.testing.assert_allclose(T_dst[:3, :3], np.eye(3), atol=1e-12)

    def test_cli_source_pos_overrides_default(self, tmp_path):
        T_src, _ = self._parse_frames(["--source-pos", "5", "6", "7"])
        np.testing.assert_allclose(T_src[:3, 3], [5.0, 6.0, 7.0])

    def test_cli_dest_rpy_overrides_default(self, tmp_path):
        T_src, T_dst = self._parse_frames(["--dest-rpy", "0", "0", str(math.pi / 2)])
        expected_R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=float)
        np.testing.assert_allclose(T_dst[:3, :3], expected_R, atol=1e-12)

    def test_config_overrides_default(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [2.0, 0.0, 0.0], "rpy": [0, 0, 0]},
            "destination": {"position": [-2.0, 0.0, 0.0], "rpy": [0, 0, 0]},
        })
        T_src, T_dst = self._parse_frames(["--config", str(cfg_path)])
        np.testing.assert_allclose(T_src[:3, 3], [2.0, 0.0, 0.0])
        np.testing.assert_allclose(T_dst[:3, 3], [-2.0, 0.0, 0.0])

    def test_cli_overrides_config(self, tmp_path):
        cfg_path = _write_config(tmp_path, {
            "source": {"position": [2.0, 0.0, 0.0], "rpy": [0, 0, 0]},
            "destination": {"position": [-2.0, 0.0, 0.0], "rpy": [0, 0, 0]},
        })
        T_src, _ = self._parse_frames(
            ["--config", str(cfg_path), "--source-pos", "9", "8", "7"]
        )
        # CLI --source-pos=9 8 7 should beat config's position=[2,0,0]
        np.testing.assert_allclose(T_src[:3, 3], [9.0, 8.0, 7.0])
