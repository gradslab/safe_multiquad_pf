r"""Shared drawing style for the project-page figures.

One palette, one type scale, one grid treatment. Figures are drawn on an opaque light plate because
the page shows them on a white card in both its themes.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Paul Tol bright, the palette the simulation logs already use for the four agents
AGENT = {"quad_A": "#4477AA", "quad_B": "#EE6677", "quad_C": "#228833", "quad_D": "#CCBB44",
         "q0": "#4477AA", "q1": "#EE6677", "q2": "#228833", "q3": "#CCBB44"}
CTRL = {"proposed": "#1F4E79", "baseline": "#EE6677", "se3": "#CC8B00"}
INK = "#12171C"
MUTED = "#5A6772"
GRID = "#DCE3E9"
PLATE = "#FFFFFF"
DANGER = "#C1121F"
OK = "#228833"
SEQ = "viridis"

BASE = {
    "figure.facecolor": PLATE, "axes.facecolor": PLATE, "savefig.facecolor": PLATE,
    "font.family": "sans-serif",
    "font.sans-serif": ["IBM Plex Sans", "DejaVu Sans", "Helvetica", "Arial"],
    "mathtext.fontset": "dejavusans",
    "text.color": INK, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.edgecolor": "#B7C2CC", "axes.linewidth": 0.9,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.7, "grid.alpha": 1.0,
    "axes.axisbelow": True,
    "axes.spines.top": False, "axes.spines.right": False,
    "legend.frameon": True, "legend.framealpha": 0.96, "legend.edgecolor": "#D5DDE4",
    "legend.facecolor": PLATE, "legend.borderpad": 0.5, "legend.handlelength": 1.9,
    "lines.solid_capstyle": "round",
}


def use(scale=1.0):
    """Apply the theme. `scale` sets the type size relative to a figure authored at 12 in wide."""
    plt.rcParams.update(BASE)
    plt.rcParams.update({
        "font.size": 13 * scale, "axes.labelsize": 13 * scale, "axes.titlesize": 14 * scale,
        "xtick.labelsize": 12 * scale, "ytick.labelsize": 12 * scale,
        "legend.fontsize": 12 * scale, "figure.titlesize": 15 * scale,
    })


def panel_title(ax, text, pad=9):
    ax.set_title(text, loc="left", pad=pad, fontweight="semibold", color=INK)


def tag(ax, letter, dx=-0.085, dy=1.035):
    ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=plt.rcParams["axes.titlesize"],
            fontweight="bold", color=INK, ha="left", va="bottom")


def finish(fig, w=None, h=None):
    if w and h:
        fig.set_size_inches(w, h)
    return fig
