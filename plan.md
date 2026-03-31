# Sheth-Uicker Visualizer — Calculation Plan

## 1. Overview

This document describes all mathematical operations and functions required to implement a Python 3D application that:

- Accepts **two coordinate frames** (each defined by a 3-D position and a 3-D orientation).
- Computes the **Sheth-Uicker parameters** L1, L2, L3, A1, A2, A3 that express the rigid-body transformation from frame 1 to frame 2.
- Renders both frames and the six-step decomposition path in an interactive 3D scene.

Measurements follow the **Z → X′ → Z′′** (ZXZ) Euler / Sheth-Uicker convention.

---

## 2. Background: Sheth-Uicker Parameters

Sheth and Uicker (1971) showed that *any* rigid-body transformation T ∈ SE(3) can be factored into a sequence of six elementary screws along / about the axes Z, X′, Z′′:

```
T = Rz(A1) · Tz(L1) · Rx(A2) · Tx(L2) · Rz(A3) · Tz(L3)
```

where:

| Parameter | Type        | Axis | Meaning                                      |
|-----------|-------------|------|----------------------------------------------|
| A1        | angle (rad) | Z    | First rotation — about the Z axis            |
| L1        | length (m)  | Z    | First translation — along the (rotated) Z axis |
| A2        | angle (rad) | X′   | Second rotation — about the intermediate X axis |
| L2        | length (m)  | X′   | Second translation — along the intermediate X axis |
| A3        | angle (rad) | Z′′  | Third rotation — about the final Z axis      |
| L3        | length (m)  | Z′′  | Third translation — along the final Z axis   |

The convention is equivalent to expressing the relative transformation as a composition of **two half-screws about Z** separated by **one half-screw about X**.

---

## 3. Coordinate Frame Representation

Each input frame is represented by:

- **Position** `p` — a 3-vector `[x, y, z]` (origin of the frame).
- **Orientation** `R` — a 3×3 rotation matrix (columns are the X, Y, Z unit vectors of the frame, expressed in the world / reference frame).

Alternatively, orientation can be supplied as:

- A **unit quaternion** `q = [qw, qx, qy, qz]`, or
- **Euler angles** in any standard convention (e.g. roll-pitch-yaw).

Conversion to a 4×4 homogeneous transformation matrix is the first step in all calculations.

---

## 4. Calculation Pipeline

### Step 1 — Build homogeneous transformation matrices

Given frame *i* with position `pᵢ` and rotation matrix `Rᵢ`:

```
       | Rᵢ   pᵢ |
Tᵢ  = |          |
       | 0    1  |
```

Functions needed:

- `rotation_matrix_from_euler(angles, convention)` → R (3×3)
- `rotation_matrix_from_quaternion(q)` → R (3×3)
- `build_homogeneous(R, p)` → T (4×4)

---

### Step 2 — Compute the relative transformation

The transformation that maps frame 1 into frame 2, expressed in the coordinate system of frame 1, is:

```
T_rel = T1⁻¹ · T2
```

Because T is a rigid-body transformation its inverse has a closed form:

```
T1⁻¹ = | R1ᵀ   −R1ᵀ·p1 |
        | 0         1   |
```

Functions needed:

- `invert_homogeneous(T)` → T_inv (4×4)
- `relative_transform(T1, T2)` → T_rel (4×4)

---

### Step 3 — Extract the rotation and translation from T_rel

```
T_rel = | R_rel   p_rel |
        | 0          1  |
```

Functions needed:

- `extract_rotation(T)` → R (3×3)
- `extract_translation(T)` → p (3-vector)

---

### Step 4 — Decompose the rotation into ZXZ Euler angles (A1, A2, A3)

The rotation matrix R_rel is decomposed as:

```
R_rel = Rz(A1) · Rx(A2) · Rz(A3)
```

**Closed-form solution:**

Given `R_rel[i][j]` (0-indexed):

```
A2 = atan2( sqrt(R[2,0]² + R[2,1]²),  R[2,2] )      # tilt (0 ≤ A2 ≤ π)

if sin(A2) ≠ 0:                                        # non-degenerate case
    A1 = atan2(  R[1,2] / sin(A2),   R[0,2] / sin(A2) )
    A3 = atan2(  R[2,1] / sin(A2),  -R[2,0] / sin(A2) )

else:                                                  # gimbal lock (A2 = 0 or π)
    # Only A1 + A3 (or A1 - A3) is determined; set A1 = 0 by convention
    A1 = 0
    if R[2,2] > 0:                                     # A2 ≈ 0
        A3 = atan2(R[1,0], R[0,0])
    else:                                              # A2 ≈ π
        A3 = atan2(-R[1,0], -R[0,0])
```

Functions needed:

- `decompose_zxz(R)` → `(A1, A2, A3)` in radians
- `rotation_zxz(A1, A2, A3)` → R (3×3) — forward computation, used to verify

---

### Step 5 — Reconstruct the intermediate frames along the decomposition path

The six elementary transformations build a "chain" of intermediate frames F0 … F6, where F0 = frame 1 and F6 = frame 2 (up to floating-point error).

```
F0 = T1
F1 = F0 · Rz(A1)         # after first rotation
F2 = F1 · Tz(L1)         # after first translation
F3 = F2 · Rx(A2)         # after second rotation
F4 = F3 · Tx(L2)         # after second translation
F5 = F4 · Rz(A3)         # after third rotation
F6 = F5 · Tz(L3)         # after third translation  (≈ T2)
```

Functions needed:

- `elementary_rotation(axis, angle)` → T (4×4) — axis ∈ {'x','y','z'}
- `elementary_translation(axis, distance)` → T (4×4) — axis ∈ {'x','y','z'}
- `decomposition_chain(T1, A1, L1, A2, L2, A3, L3)` → list of 7 frames (4×4 matrices)

---

### Step 6 — Solve for the translation parameters L1, L2, L3

Once A1, A2, A3 are known, the three translations can be solved from the position part of T_rel.

The full transformation expanded gives the position:

```
p_rel = Rz(A1) · ( L1·ẑ + Rx(A2)·( L2·x̂ + Rz(A3)·L3·ẑ ) )
```

Rearranging (dot product with the appropriate unit vectors of each intermediate frame):

```
L1 = ( R_A1ᵀ · p_rel ) · ẑ  − L3·( R_A1ᵀ·R_A2·R_A3·ẑ )·ẑ  − L2·( R_A1ᵀ·R_A2·x̂ )·ẑ

-- more compactly solved by propagating left-to-right and isolating each component:

p1 = Rz(A1)ᵀ · p_rel                         # express p_rel in intermediate frame after A1
L1 = p1[2] − L2·(Rx(A2)·x̂)[2] − L3·(Rx(A2)·Rz(A3)·ẑ)[2]

p2 = Rx(A2)ᵀ · ( p1 − L1·ẑ )               # remaining displacement after L1·ẑ
L2 = p2[0] − L3·(Rz(A3)·ẑ)[0]

p3 = Rz(A3)ᵀ · ( p2 − L2·x̂ )               # remaining displacement after L2·x̂
L3 = p3[2]
```

> **Alternative (direct numerical approach):** Build the symbolic equation system, substitute the known angles, and solve the resulting 3×3 linear system for [L1, L2, L3]. This is more robust when angles are close to gimbal-lock.

Functions needed:

- `solve_translations(R_rel, p_rel, A1, A2, A3)` → `(L1, L2, L3)`

**Canonicalization (bijection constraints):**

After extracting the six parameters, normalize them to ensure a **unique** representation for any spatial relationship (bijection). The implementation will enforce:

- **L1 ∈ ℝ**
- **L2 ≥ 0**
- **L3 ∈ ℝ**
- **A1 ∈ [0, 2π[**
- **A2 ∈ (−π, π)**
- **A3 ∈ [0, 2π[**
- **(A2 = 0) ⇒ (L3 = 0)**
- **(L2 = 0 ∧ A2 = 0) ⇒ (A3 = 0)**

This step ensures the parameter set is canonical and eliminates ambiguous solutions that represent the same transform.

---

### Step 7 — Validation

Reconstruct T from the six parameters and verify it matches T_rel:

```
T_reconstructed = Rz(A1)·Tz(L1)·Rx(A2)·Tx(L2)·Rz(A3)·Tz(L3)
error = || T_reconstructed − T_rel ||_F
```

Functions needed:

- `reconstruct_transform(A1, L1, A2, L2, A3, L3)` → T (4×4)
- `frobenius_error(T1, T2)` → scalar

---

## 5. Complete Function List

```
# ── I/O helpers ───────────────────────────────────────────────────────────────
rotation_matrix_from_euler(angles: array-like, convention: str) -> ndarray[3,3]
rotation_matrix_from_quaternion(q: array-like) -> ndarray[3,3]
build_homogeneous(R: ndarray, p: array-like) -> ndarray[4,4]

# ── Core linear algebra ───────────────────────────────────────────────────────
invert_homogeneous(T: ndarray) -> ndarray[4,4]
relative_transform(T1: ndarray, T2: ndarray) -> ndarray[4,4]
extract_rotation(T: ndarray) -> ndarray[3,3]
extract_translation(T: ndarray) -> ndarray[3]
elementary_rotation(axis: str, angle: float) -> ndarray[4,4]
elementary_translation(axis: str, distance: float) -> ndarray[4,4]

# ── Sheth-Uicker decomposition ────────────────────────────────────────────────
decompose_zxz(R: ndarray) -> tuple[float, float, float]        # (A1, A2, A3) rad
rotation_zxz(A1, A2, A3: float) -> ndarray[3,3]
solve_translations(R_rel, p_rel, A1, A2, A3: float) -> tuple[float, float, float]  # (L1, L2, L3)
canonicalize_parameters(A1: float, A2: float, A3: float, L1: float, L2: float, L3: float) -> tuple[float, float, float, float, float, float]  # returns (A1, A2, A3, L1, L2, L3)
compute_sheth_uicker(T1: ndarray, T2: ndarray) -> dict          # returns all 6 params

# ── Reconstruction / validation ───────────────────────────────────────────────
reconstruct_transform(A1, L1, A2, L2, A3, L3: float) -> ndarray[4,4]
frobenius_error(T_a, T_b: ndarray) -> float
decomposition_chain(T1, A1, L1, A2, L2, A3, L3) -> list[ndarray[4,4]]  # 7 frames

# ── Visualisation ─────────────────────────────────────────────────────────────
draw_frame(ax, T: ndarray, scale: float, label: str)            # draws X/Y/Z arrows
draw_decomposition_path(ax, chain: list[ndarray])               # draws intermediate frames + connecting segments
render_scene(T1, T2, params: dict)                              # main entry-point for 3D plot
```

---

## 6. Top-level Algorithm (Pseudocode)

```python
def compute_and_visualize(frame1, frame2):
    # 1. Build homogeneous matrices
    T1 = build_homogeneous(frame1.R, frame1.p)
    T2 = build_homogeneous(frame2.R, frame2.p)

    # 2. Relative transformation
    T_rel = relative_transform(T1, T2)
    R_rel = extract_rotation(T_rel)
    p_rel = extract_translation(T_rel)

    # 3. Angle decomposition (ZXZ)
    A1, A2, A3 = decompose_zxz(R_rel)

    # 4. Translation extraction
    L1, L2, L3 = solve_translations(R_rel, p_rel, A1, A2, A3)

    # 4b. Canonicalize parameters for bijection
    A1, A2, A3, L1, L2, L3 = canonicalize_parameters(A1, A2, A3, L1, L2, L3)

    # 5. Validate
    T_check = reconstruct_transform(A1, L1, A2, L2, A3, L3)
    err = frobenius_error(T_rel, T_check)
    assert err < 1e-10, f"Reconstruction error too large: {err}"

    # 6. Build decomposition chain for visualisation
    chain = decomposition_chain(T1, A1, L1, A2, L2, A3, L3)

    # 7. Render
    params = dict(A1=A1, L1=L1, A2=A2, L2=L2, A3=A3, L3=L3)
    render_scene(T1, T2, params, chain)

    return params
```

---

## 7. Visualisation Plan

The 3D scene will be built with **Matplotlib (mpl_toolkits.mplot3d)** or **Plotly** (interactive):

| Element | Visual representation |
|---|---|
| Frame 1 (input) | RGB triad (R=X, G=Y, B=Z), label "Frame 1" |
| Frame 2 (input) | RGB triad, label "Frame 2" |
| Intermediate frames F1 … F5 | Lighter RGB triads, label "Fᵢ" |
| Translation segments (L1, L2, L3) | Dashed line along the local axis, annotated with value |
| Rotation arcs (A1, A2, A3) | Circular arc around the rotation axis, annotated with value in degrees |
| Decomposition path | Thin solid polyline connecting F0 → F6 origins |

A text panel beside the plot will display the six numerical parameter values.

---

## 8. Edge Cases and Numerical Considerations

| Situation | Handling |
|---|---|
| A2 ≈ 0 or A2 ≈ π (gimbal lock) | Set A1 = 0 by convention; solve for A3 from the combined angle |
| Both frames are identical | All six parameters are 0; render coincident frames |
| Pure translation (R_rel = I) | A1 = A2 = A3 = 0; L1, L2, L3 determined by dot products |
| Pure rotation (p_rel = 0) | L1 = L2 = L3 = 0; angles from ZXZ decomposition |
| Very large / very small distances | Use `numpy` double precision throughout; normalise visualisation scale |

---

## 9. Recommended Python Dependencies

| Library | Purpose |
|---|---|
| `numpy` | All matrix / vector arithmetic |
| `scipy.spatial.transform.Rotation` | Input orientation parsing (quaternion, Euler, etc.) |
| `matplotlib` | 3D visualisation (baseline, always available) |
| `plotly` *(optional)* | Interactive 3D visualisation |

---

## 10. Suggested File Structure

```
py-sheth-uicker-visualizer/
├── plan.md                    ← this file
├── sheth_uicker/
│   ├── __init__.py
│   ├── transforms.py          ← Steps 1-3 (homogeneous matrices, inversion)
│   ├── decomposition.py       ← Steps 4-6 (ZXZ decomposition, translations)
│   ├── validation.py          ← Step 7 (reconstruction, error check)
│   └── visualisation.py       ← Step 8 (Matplotlib / Plotly scene)
├── main.py                    ← CLI entry-point calling compute_and_visualize()
└── tests/
    ├── test_transforms.py
    ├── test_decomposition.py
    └── test_validation.py
```
