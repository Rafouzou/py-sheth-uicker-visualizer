"""3-D scene rendering utilities.

Implements:
  - draw_frame            : draws a single coordinate frame triad (X/Y/Z arrows + label)
  - draw_decomposition_path: draws the six-step decomposition chain
  - render_scene          : opens a Matplotlib 3-D window with the full decomposition
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 – registers the 3D projection


# ── colour palettes ────────────────────────────────────────────────────────────
# Each frame is drawn with R=X, G=Y, B=Z arrows (standard robotics colouring).
_AXIS_COLORS = ("red", "green", "blue")
_AXIS_LABELS = ("X", "Y", "Z")

# Intermediate frames use lighter colours
_INTERMEDIATE_ALPHA = 0.45
_PATH_COLOR = "dimgray"


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


def draw_decomposition_path(
    ax: "Axes3D",
    chain: list,
    scale: float = 0.3,
) -> None:
    """Draw the six-step decomposition chain on a 3-D axes.

    The chain contains 7 frames F0 … F6.  Intermediate frames F1 … F5 are
    drawn as lighter triads.  The path connecting the origins F0→F1→…→F6 is
    drawn as a dashed polyline.

    Parameters
    ----------
    ax    : mpl_toolkits.mplot3d.Axes3D
    chain : list of 7 ndarray (4, 4) — from :func:`~sheth_uicker.validation.decomposition_chain`
    scale : arrow length for intermediate frames
    """
    # Dashed polyline connecting all frame origins
    origins = np.array([F[:3, 3] for F in chain])
    ax.plot(
        origins[:, 0], origins[:, 1], origins[:, 2],
        linestyle="--",
        color=_PATH_COLOR,
        linewidth=1.2,
        alpha=0.7,
        zorder=1,
    )

    # Draw intermediate frames F1 … F5 (not source F0 nor destination F6)
    for i, F in enumerate(chain[1:-1], start=1):
        draw_frame(ax, F, scale=scale, label=f"F{i}", alpha=_INTERMEDIATE_ALPHA)


def render_scene(
    T_source: np.ndarray,
    T_dest: np.ndarray,
    *,
    params: dict | None = None,
    chain: list | None = None,
    frame_scale: float = 0.5,
    source_label: str = "Frame 1",
    dest_label: str = "Frame 2",
) -> None:
    """Open an interactive Matplotlib 3-D window showing the Sheth-Uicker decomposition.

    Parameters
    ----------
    T_source     : (4, 4) homogeneous transform for the source frame
    T_dest       : (4, 4) homogeneous transform for the destination frame
    params       : dict with keys A1, L1, A2, L2, A3, L3 (optional)
    chain        : list of 7 frames from decomposition_chain (optional)
    frame_scale  : length of the axis arrows for source/destination frames
    source_label : text label for the source frame
    dest_label   : text label for the destination frame
    """
    if params is not None and chain is not None:
        fig = plt.figure(figsize=(12, 7))
        ax: Axes3D = fig.add_subplot(121, projection="3d")
    else:
        fig = plt.figure(figsize=(8, 7))
        ax: Axes3D = fig.add_subplot(111, projection="3d")

    draw_frame(ax, T_source, scale=frame_scale, label=source_label)
    draw_frame(ax, T_dest,   scale=frame_scale, label=dest_label)

    if chain is not None:
        draw_decomposition_path(ax, chain, scale=frame_scale * 0.6)

    # ── axes appearance ────────────────────────────────────────────────────────
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Sheth-Uicker Visualizer")

    _set_axes_equal(ax)
    ax.set_box_aspect([1, 1, 1])
    ax.view_init(elev=20, azim=45)

    # ── parameter panel ────────────────────────────────────────────────────────
    if params is not None:
        ax2 = fig.add_subplot(122)
        ax2.axis("off")

        import math
        lines = [
            "Sheth-Uicker Parameters",
            "",
            f"  A1 = {math.degrees(params['A1']):8.3f} °",
            f"  L1 = {params['L1']:8.4f}",
            f"  A12 = {math.degrees(params['A2']):8.3f} °",
            f"  L12 = {params['L2']:8.4f}",
            f"  A2 = {math.degrees(params['A3']):8.3f} °",
            f"  L2 = {params['L3']:8.4f}",
        ]
        ax2.text(
            0.05, 0.55, "\n".join(lines),
            transform=ax2.transAxes,
            fontsize=12,
            verticalalignment="center",
            fontfamily="monospace",
            bbox=dict(boxstyle="round", facecolor="lightyellow", alpha=0.8),
        )

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
