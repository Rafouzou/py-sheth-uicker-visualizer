"""3-D scene rendering utilities.

Currently implements:
  - draw_frame  : draws a single coordinate frame triad (X/Y/Z arrows + label)
  - render_scene: opens a Matplotlib 3-D window with source and destination frames
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers the 3D projection


# ── colour palettes ────────────────────────────────────────────────────────────
# Each frame is drawn with R=X, G=Y, B=Z arrows (standard robotics colouring).
_AXIS_COLORS = ("red", "green", "blue")
_AXIS_LABELS = ("X", "Y", "Z")


def draw_frame(
    ax: "Axes3D",
    T: np.ndarray,
    scale: float = 0.5,
    label: str = "",
    alpha: float = 1.0,
) -> None:
    """Draw a coordinate frame triad on a 3-D Matplotlib axes.

    Parameters
    ----------
    ax    : mpl_toolkits.mplot3d.Axes3D
    T     : ndarray (4, 4)  homogeneous transform — defines the frame pose
    scale : arrow length in world units
    label : text label drawn at the frame origin
    alpha : transparency (1 = opaque)
    """
    origin = T[:3, 3]
    R = T[:3, :3]

    for i, (color, axis_label) in enumerate(zip(_AXIS_COLORS, _AXIS_LABELS)):
        direction = R[:, i] * scale
        ax.quiver(
            origin[0], origin[1], origin[2],
            direction[0], direction[1], direction[2],
            color=color,
            alpha=alpha,
            linewidth=2,
            arrow_length_ratio=0.2,
        )
        tip = origin + direction * 1.15
        ax.text(
            tip[0], tip[1], tip[2],
            axis_label,
            color=color,
            fontsize=7,
            ha="center",
            va="center",
        )

    if label:
        ax.text(
            origin[0], origin[1], origin[2] + scale * 1.3,
            label,
            fontsize=9,
            fontweight="bold",
            ha="center",
            va="bottom",
        )


def render_scene(
    T_source: np.ndarray,
    T_dest: np.ndarray,
    *,
    frame_scale: float = 0.5,
    source_label: str = "Source",
    dest_label: str = "Destination",
) -> None:
    """Open an interactive Matplotlib 3-D window showing two coordinate frames.

    Parameters
    ----------
    T_source     : (4, 4) homogeneous transform for the source frame
    T_dest       : (4, 4) homogeneous transform for the destination frame
    frame_scale  : length of the axis arrows
    source_label : text label for the source frame
    dest_label   : text label for the destination frame
    """
    fig = plt.figure(figsize=(8, 7))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    draw_frame(ax, T_source, scale=frame_scale, label=source_label)
    draw_frame(ax, T_dest, scale=frame_scale, label=dest_label)

    # ── axes appearance ────────────────────────────────────────────────────────
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sheth-Uicker Visualizer")

    # Make equal aspect ratio so the triads are not distorted.
    _set_axes_equal(ax)

    # Z is "up": set a sensible default viewing angle (elevation 20°, azimuth 45°)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()
    plt.show()


# ── helpers ────────────────────────────────────────────────────────────────────

def _set_axes_equal(ax: "Axes3D") -> None:
    """Force equal scaling on all three axes of a 3-D Matplotlib plot."""
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    centers = limits.mean(axis=1)
    half_range = (limits[:, 1] - limits[:, 0]).max() / 2.0

    ax.set_xlim3d(centers[0] - half_range, centers[0] + half_range)
    ax.set_ylim3d(centers[1] - half_range, centers[1] + half_range)
    ax.set_zlim3d(centers[2] - half_range, centers[2] + half_range)
