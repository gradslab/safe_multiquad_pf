r"""Path-following result for N=4: a 3D view and a top view side by side, all three controllers.

Shows the whole run including the convergence from the off-path start, so the reader can see the
agents reach their circles and then stay on them while the two cascades are pushed off.
Proposed solid, TFL + safety filter dashed, SE(3) + safety filter dotted; assigned paths in grey.

Usage: python3 experiments/fig_track3d.py --tag _cs1p4
"""
import argparse, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp")); sys.path.insert(0, PKG)
import paths as P
import qc_figures
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
# Also write straight into the paper's figures/ directory. Copying by hand meant the paper embedded a
# stale figure whenever a fix landed after the copy; writing both places removes that failure mode.
PAPER_FIG = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))
SPACER = 0.10
COL = {"quad_A": "#4477AA", "quad_B": "#EE6677", "quad_C": "#228833", "quad_D": "#CCBB44"}
CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}
CTRL = [("", "-", 1.30, "proposed"),
        ("_baseline", "--", 1.05, "TFL + safety filter"),
        ("_se3", ":", 1.05, "[31] + safety filter")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_cs1p4"); ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--out", default="F2_track3d_N4")
    ap.add_argument("--paper", action="store_true", help="author at the printed width, 7 pt type")
    ap.add_argument("--spacer", type=float, default=0.10,
                    help="width of the empty column between the two panels, as a fraction")
    a = ap.parse_args()
    dg = np.load(os.path.join(RES, f"circles_N{a.N}{a.tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in dg["names"]]
    runs = []
    for sfx, ls, lw, _ in CTRL:
        f = os.path.join(RES, f"circles_N{a.N}{sfx}{a.tag}.npz")
        if os.path.exists(f):
            S = np.load(f); runs.append(({n: S[n] for n in names}, ls, lw))

    # PAPER: authored at the width LaTeX prints it at, so the type size on this canvas is the type
    # size on the page. At the old 9.4 in canvas every label arrived at about 3 pt.
    PAPER = a.paper
    global SPACER
    SPACER = a.spacer
    plt.rcParams.update({"font.size": 7 if PAPER else 9, "legend.frameon": False,
                         "axes.linewidth": 0.6 if PAPER else 0.8,
                         "xtick.major.width": 0.6 if PAPER else 0.8,
                         "ytick.major.width": 0.6 if PAPER else 0.8})
    fig = plt.figure(figsize=(3.40, 2.15) if PAPER else (9.4, 4.0))
    # mplot3d always hangs the z axis furniture off the right of its box, which is exactly where the
    # top view's own y axis lives. An empty middle column reserves that space, so the two panels
    # cannot collide however far the projected labels reach.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.02, 0.30, 1.0] if PAPER else [1.15, 0.02, 1.0],
                          wspace=0.06, left=0.015, right=0.90 if PAPER else 0.985,
                          top=0.885 if PAPER else 0.88, bottom=0.31 if PAPER else 0.10)
    ax3 = fig.add_subplot(gs[0], projection="3d")
    axt = fig.add_subplot(gs[2])
    # zoom > 1 makes mplot3d draw its box larger than the cell, so the z axis furniture spills past
    # the spacer column no matter how wide it is. Keep the box inside its cell.
    ax3.set_box_aspect((1, 1, 0.45), zoom=1.16 if PAPER else 1.18)

    qq = np.linspace(0, 2 * np.pi, 400)
    for n in names:
        d = np.array([P.lifted_circle(n, *CEN[n], R=1.5).sig(q, 0) for q in qq])
        ax3.plot(d[:, 0], d[:, 1], d[:, 2], color="0.55", lw=0.7, ls=(0, (1, 2)), zorder=1)
        axt.plot(d[:, 0], d[:, 1], color="0.55", lw=0.7, ls=(0, (1, 2)), zorder=1)

    for st, ls, lw in runs:
        for n in names:
            X = st[n]
            _lw = 0.75 * lw if PAPER else lw
            ax3.plot(X[:, 6], X[:, 7], X[:, 8], color=COL[n], lw=_lw, ls=ls, zorder=3)
            axt.plot(X[:, 6], X[:, 7], color=COL[n], lw=_lw, ls=ls, zorder=3)
    # off-path starts, marked once
    for n in names:
        X = runs[0][0][n]
        _s = 12 if PAPER else 26
        ax3.scatter(X[0, 6], X[0, 7], X[0, 8], color=COL[n], marker="o", s=_s,
                    depthshade=False, zorder=5, edgecolor="k", linewidth=0.4)
        axt.scatter(X[0, 6], X[0, 7], color=COL[n], marker="o", s=_s, zorder=5,
                    edgecolor="k", linewidth=0.4)

    ax3.set_xlabel("$x$ [m]", labelpad=-6 if PAPER else -4)
    ax3.set_ylabel("$y$ [m]", labelpad=-6 if PAPER else -4)
    ax3.set_zlabel("$z$ [m]", labelpad=-7 if PAPER else -6)
    ax3.tick_params(labelsize=5.5 if PAPER else 7, pad=-3 if PAPER else -2)
    if PAPER:
        from matplotlib.ticker import MaxNLocator
        for _ax in (ax3.xaxis, ax3.yaxis, ax3.zaxis):
            _ax.set_major_locator(MaxNLocator(nbins=3))
    ax3.set_title("(a) three-dimensional view", fontsize=7 if PAPER else 10, pad=2)
    if PAPER:
        # the 3d panel puts its z ticks and label on its right; the top view normally puts its y
        # ticks and label on its left, so the two face each other across the gap. Moving the top
        # view's y axis to its own right edge removes the conflict instead of negotiating it.
        axt.yaxis.set_label_position("right")
        axt.yaxis.tick_right()
    axt.set_aspect("equal", adjustable="box")
    axt.set_xlabel("$x$ [m]", labelpad=1); axt.set_ylabel("$y$ [m]", labelpad=1)
    axt.set_title("(b) top view", fontsize=7 if PAPER else 10, pad=5)
    axt.tick_params(labelsize=6 if PAPER else 8)
    axt.grid(alpha=0.25, lw=0.4 if PAPER else 0.5)
    if PAPER:
        from matplotlib.ticker import MaxNLocator
        axt.xaxis.set_major_locator(MaxNLocator(nbins=4))
        axt.yaxis.set_major_locator(MaxNLocator(nbins=4))

    handles = [Line2D([], [], color="0.3", lw=lw, ls=ls, label=lab) for _, ls, lw, lab in CTRL]
    if True:
        handles.append(Line2D([], [], color="0.55", lw=0.7, ls=(0, (1, 2)), label="assigned path"))
        handles.append(Line2D([], [], color="0.3", marker="o", ls="none", ms=4.5,
                              label="off-path start"))
    # at column width a five entry legend needs two rows, so it goes under the panels where it
    # cannot reach the titles
    fig.legend(handles=handles, ncol=3 if PAPER else 5,
               loc="lower center" if PAPER else "upper center",
               bbox_to_anchor=(0.5, -0.015) if PAPER else (0.5, 1.02),
               fontsize=6.2 if PAPER else 8.5, handlelength=1.9 if PAPER else 2.4,
               columnspacing=1.0 if PAPER else 1.4)
    if PAPER:
        sys.path.insert(0, os.path.join(HERE, "web"))
        import webqc
        issues = webqc.gate_print(fig, a.out, printed_in=3.40, min_pt=5.5, require_labels=False)
    else:
        issues = qc_figures.check(fig, a.out)
    for e in (".pdf", ".png", ".eps"):
        fig.savefig(os.path.join(FIG, a.out + e), bbox_inches="tight", pad_inches=0.01,
                    **({"dpi": 200} if e == ".png" else {}))
    if os.path.isdir(PAPER_FIG):
        fig.savefig(os.path.join(PAPER_FIG, a.out + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {a.out}.pdf/.png/.eps")
    return 1 if issues else 0


if __name__ == "__main__":
    sys.exit(main())
