"""Entry-point for the Sheth-Uicker visualizer.

Default behaviour
-----------------
Opens a 3-D window containing two coordinate frames:
  - Source frame at x = +1, y = 0, z = 0 (identity orientation)
  - Destination frame at x = -1, y = 0, z = 0 (identity orientation)

Z is world-up.

Usage
-----
    python main.py
"""

import numpy as np

from sheth_uicker.transforms import build_homogeneous, identity_frame
from sheth_uicker.visualisation import render_scene


def _default_source_frame() -> np.ndarray:
    """Source frame at x = +1, identity rotation."""
    return build_homogeneous(np.eye(3), [1.0, 0.0, 0.0])


def _default_dest_frame() -> np.ndarray:
    """Destination frame at x = -1, identity rotation."""
    return build_homogeneous(np.eye(3), [-1.0, 0.0, 0.0])


def main() -> None:
    T_source = _default_source_frame()
    T_dest = _default_dest_frame()

    print("Source frame T:")
    print(T_source)
    print("\nDestination frame T:")
    print(T_dest)

    render_scene(T_source, T_dest)


if __name__ == "__main__":
    main()
