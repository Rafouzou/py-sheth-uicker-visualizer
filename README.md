# py-sheth-uicker-visualizer

Python 3D application to visualize the spatial relationship between two sets of axes using Sheth-Uicker parameters.

## Quick start

```bash
pip install -r requirements.txt
python main.py
```

This opens a 3-D Matplotlib window with two coordinate frames:

- **Source** frame at position (+1, 0, 0), identity orientation
- **Destination** frame at position (−1, 0, 0), identity orientation

---

## CLI usage

Frames can be positioned and oriented freely via command-line flags.

```
python main.py [--config FILE]
               [--source-pos X Y Z] [--source-rpy ROLL PITCH YAW]
               [--dest-pos   X Y Z] [--dest-rpy   ROLL PITCH YAW]
```

| Flag | Description | Default |
|---|---|---|
| `--config FILE` | Load poses from a JSON file (see below) | — |
| `--source-pos X Y Z` | Source frame position in world coordinates | `1 0 0` |
| `--source-rpy R P Y` | Source frame roll/pitch/yaw (radians) | `0 0 0` |
| `--dest-pos X Y Z` | Destination frame position in world coordinates | `-1 0 0` |
| `--dest-rpy R P Y` | Destination frame roll/pitch/yaw (radians) | `0 0 0` |

**Precedence:** hardcoded defaults → JSON config → CLI flags (highest priority).

### Examples

```bash
# Source rotated 90° about Z, destination rotated 45° about X
python main.py \
  --source-pos 1 0 0 --source-rpy 0 0 1.5708 \
  --dest-pos  -1 0 0 --dest-rpy   0.7854 0 0

# Load from config, then override destination position
python main.py --config my_config.json --dest-pos 0 2 0
```

---

## JSON config file

Pass a JSON file with `--config` to describe both frame poses.

```json
{
  "source": {
    "position": [1.0, 0.0, 0.0],
    "rpy": [0.0, 0.0, 0.0]
  },
  "destination": {
    "position": [-1.0, 0.0, 0.0],
    "rotation_matrix": [
      [1, 0, 0],
      [0, 1, 0],
      [0, 0, 1]
    ]
  }
}
```

### Orientation keys

Each frame supports two mutually exclusive orientation keys:

| Key | Format | Description |
|---|---|---|
| `rpy` | `[roll, pitch, yaw]` (radians) | Converted to a rotation matrix using the ZYX convention |
| `rotation_matrix` | 3×3 nested list | Used directly as the rotation matrix |

**When both `rpy` and `rotation_matrix` are present, `rotation_matrix` takes precedence and `rpy` is ignored.**

If neither key is provided, the frame defaults to an identity (zero) rotation.

### Validation

| Field | Requirement |
|---|---|
| `position` | List of exactly 3 numbers |
| `rpy` | List of exactly 3 numbers (radians) |
| `rotation_matrix` | 3×3 nested list of numbers |

Clear error messages are printed for shape/type violations.

---

## Orientation convention

All orientation inputs (CLI `--source-rpy` / `--dest-rpy` and JSON `rpy`) use the **intrinsic ZYX** convention, equivalent to **extrinsic XYZ**:

```
R = Rz(yaw) @ Ry(pitch) @ Rx(roll)
```

Rotations are composed as:
1. **Roll** — rotate about the X-axis
2. **Pitch** — rotate about the (new) Y-axis
3. **Yaw** — rotate about the (new) Z-axis

All angles are in **radians**.

---

## Running tests

```bash
pip install pytest
python -m pytest tests/ -v
```
