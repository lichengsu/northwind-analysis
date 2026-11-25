"""Shared matplotlib style and color tokens for the report figures."""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SECONDARY = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
NEUTRAL = "#b5b3ab"


def apply():
    """Set global matplotlib defaults for the report's charts.

    Args:
        None.

    Returns:
        None. Updates ``matplotlib.pyplot.rcParams`` in place.
    """
    plt.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"],
        "font.size": 10,
        "text.color": INK,
        "axes.labelcolor": INK_SECONDARY,
        "axes.edgecolor": AXIS,
        "axes.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "axes.axisbelow": True,
        "grid.color": GRID,
        "grid.linewidth": 0.6,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": INK_SECONDARY,
        "ytick.labelcolor": INK_SECONDARY,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlelocation": "left",
        "axes.titlepad": 12,
        "legend.frameon": False,
    })


def save(fig, path):
    """Write a figure to a 200-dpi image file and close it.

    Args:
        fig (matplotlib.figure.Figure): Figure to save.
        path (str | pathlib.Path): Output image path; overwritten if
            present.

    Returns:
        None. The figure is closed and cannot be drawn again.
    """
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
