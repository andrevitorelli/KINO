"""Shared plotting style for the KINO notebooks (matplotlib).

Follows a fixed-order categorical palette validated for colour-vision
deficiency, thin marks, recessive grids and no chart junk. Import with
``from kino_style import *`` (the notebooks run from ``notebooks/``).
"""

import matplotlib.pyplot as plt

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
GRID = "#e6e5e1"

#: categorical slots in fixed order (blue, orange, aqua, yellow, magenta)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA]
SEQ_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIV_MID = "#f0efec"

plt.rcParams.update(
    {
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": INK2,
        "axes.labelcolor": INK2,
        "text.color": INK,
        "xtick.color": INK2,
        "ytick.color": INK2,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "lines.linewidth": 2.0,
        "lines.markersize": 7,
        "axes.prop_cycle": plt.cycler(color=CATEGORICAL),
        "legend.frameon": False,
        "legend.fontsize": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "font.size": 10,
        "figure.dpi": 110,
    }
)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left"):
    """Direct label at the end of a line, in ink (not the series colour)."""
    ax.annotate(
        text, (x, y), xytext=(dx, dy), textcoords="offset points", color=INK2, ha=ha, fontsize=9
    )
    ax.plot([x], [y], "o", color=color, markersize=5)
