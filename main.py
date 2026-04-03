"""Entry-point for the Sheth-Uicker visualizer.

Default behaviour
-----------------
Opens a 3-D window containing two coordinate frames:
  - Source frame at x = +1, y = 0, z = 0 (identity orientation)
  - Destination frame at x = -1, y = 0, z = 0 (identity orientation)

Z is world-up.

Usage
-----
    # Default (hardcoded frames)
    python main.py

    # Override source/destination via CLI flags
    python main.py --source-pos 1 0 0 --source-rpy 0 0 1.5708 \\
                   --dest-pos -1 0 0 --dest-rpy 0 0 0

    # Load poses from a JSON config file
    python main.py --config path/to/config.json

    # JSON config + CLI override (CLI wins for any flag provided)
    python main.py --config path/to/config.json --dest-rpy 0 0 0.5

Orientation convention
----------------------
Roll/pitch/yaw (RPY) uses **intrinsic ZYX** order (equivalent to extrinsic XYZ):

    R = Rz(yaw) @ Ry(pitch) @ Rx(roll)

All angles are in **radians**.
"""

import argparse
import sys
from typing import List, Optional

import numpy as np

from sheth_uicker.config import load_config
from sheth_uicker.decomposition import compute_sheth_uicker
from sheth_uicker.transforms import build_homogeneous, rpy_to_matrix
from sheth_uicker.validation import decomposition_chain, frobenius_error, reconstruct_transform
from sheth_uicker.visualisation import render_scene

# ── Defaults ───────────────────────────────────────────────────────────────────
_DEFAULT_SOURCE_POS = [1.0, 0.0, 0.0]
_DEFAULT_SOURCE_RPY = [0.0, 0.0, 0.0]
_DEFAULT_DEST_POS   = [-1.0, 0.0, 0.0]
_DEFAULT_DEST_RPY   = [0.0, 0.0, 0.0]


# ── CLI ────────────────────────────────────────────────────────────────────────

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python main.py",
        description=(
            "Sheth-Uicker visualizer: display two 3-D coordinate frames.\n\n"
            "Orientation convention: intrinsic ZYX RPY — R = Rz(yaw) @ Ry(pitch) @ Rx(roll).\n"
            "All angles are in radians."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    p.add_argument(
        "--config",
        metavar="FILE",
        help="Path to a JSON config file describing source and destination frame poses. "
             "CLI flags override any values loaded from the config.",
    )

    # Source frame
    p.add_argument(
        "--source-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Source frame position in world coordinates (default: 1 0 0).",
    )
    p.add_argument(
        "--source-rpy",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Source frame orientation as roll/pitch/yaw in radians (default: 0 0 0). "
             "Uses intrinsic ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).",
    )

    # Destination frame
    p.add_argument(
        "--dest-pos",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Destination frame position in world coordinates (default: -1 0 0).",
    )
    p.add_argument(
        "--dest-rpy",
        nargs=3,
        type=float,
        metavar=("ROLL", "PITCH", "YAW"),
        help="Destination frame orientation as roll/pitch/yaw in radians (default: 0 0 0). "
             "Uses intrinsic ZYX convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).",
    )

    return p


# ── frame construction ─────────────────────────────────────────────────────────

def _make_frame(pos: List[float], rpy: List[float]) -> np.ndarray:
    R = rpy_to_matrix(rpy[0], rpy[1], rpy[2])
    return build_homogeneous(R, pos)


# ── main ───────────────────────────────────────────────────────────────────────

def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    # 1. Start from hardcoded defaults.
    source_pos = list(_DEFAULT_SOURCE_POS)
    dest_pos   = list(_DEFAULT_DEST_POS)
    src_R: np.ndarray = rpy_to_matrix(*_DEFAULT_SOURCE_RPY)
    dst_R: np.ndarray = rpy_to_matrix(*_DEFAULT_DEST_RPY)

    # 2. Override with config file values (if provided).
    if args.config:
        try:
            cfg = load_config(args.config)
        except (FileNotFoundError, ValueError) as exc:
            print(f"Error loading config: {exc}", file=sys.stderr)
            sys.exit(1)

        source_pos = cfg.source.position.tolist()
        dest_pos   = cfg.destination.position.tolist()
        src_R = cfg.source.rotation
        dst_R = cfg.destination.rotation

    # 3. Apply CLI overrides (CLI always wins).
    if args.source_pos is not None:
        source_pos = args.source_pos
    if args.dest_pos is not None:
        dest_pos = args.dest_pos
    if args.source_rpy is not None:
        src_R = rpy_to_matrix(*args.source_rpy)
    if args.dest_rpy is not None:
        dst_R = rpy_to_matrix(*args.dest_rpy)

    T_source = build_homogeneous(src_R, source_pos)
    T_dest   = build_homogeneous(dst_R, dest_pos)

    print("Source frame T:")
    print(T_source)
    print("\nDestination frame T:")
    print(T_dest)

    # Compute Sheth-Uicker decomposition
    params = compute_sheth_uicker(T_source, T_dest)

    import math
    print("\nSheth-Uicker Parameters:")
    print(f"  A1 = {math.degrees(params['A1']):.4f} deg    L1 = {params['L1']:.6f}")
    print(f"  A12 = {math.degrees(params['A2']):.4f} deg   L12 = {params['L2']:.6f}")
    print(f"  A2 = {math.degrees(params['A3']):.4f} deg    L2 = {params['L3']:.6f}")

    # Validate reconstruction
    T_rel = reconstruct_transform(
        params["A1"], params["L1"],
        params["A2"], params["L2"],
        params["A3"], params["L3"],
    )
    from sheth_uicker.transforms import relative_transform
    T_rel_ref = relative_transform(T_source, T_dest)
    err = frobenius_error(T_rel_ref, T_rel)
    print(f"\nReconstruction Frobenius error: {err:.2e}")
    if err > 1e-8:
        print(
            f"WARNING: reconstruction error {err:.2e} exceeds expected threshold "
            "(1e-8). The decomposition may be inaccurate.",
            file=sys.stderr,
        )

    # Build decomposition chain for visualisation
    chain = decomposition_chain(
        T_source,
        params["A1"], params["L1"],
        params["A2"], params["L2"],
        params["A3"], params["L3"],
    )

    render_scene(T_source, T_dest, params=params, chain=chain)


if __name__ == "__main__":
    main()
