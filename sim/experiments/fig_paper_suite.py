r"""The manuscript and project-page figures, from the tagged production runs.

  pf_circ_n4.eps               3D + top view, N=4 circles: proposed vs the two cascades + paths
  pf_sin_n2.eps                3D (top) + top view (bottom), N=2 sinusoids
  fig_paper_4var_combined.eps  2 rows (circles, sine) x 5 cols (transverse, altitude, speed, yaw,
                               min pairwise distance), three controllers overlaid

Everything is computed from the raw state logs so all controllers are measured identically.
Line convention: proposed solid, TFL cascade dashed, SE(3) cascade dotted; per-agent colors.

Usage: python3 experiments/fig_paper_suite.py [--tag _paper] [--outdir ../results/figs]
"""
import argparse
import itertools
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src", "tflqp")))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import scenario_circles as circ
import scenario_sine as sine

COLORS = {"quad_A": "#4477AA", "quad_B": "#EE6677", "quad_C": "#228833", "quad_D": "#CCBB44"}
STYLE = {"proposed": ("-", 1.4), "baseline": ("--", 1.0), "se3": (":", 1.2)}
LABEL = {"proposed": "proposed", "baseline": "TFL [17] + filter", "se3": "SE(3) [25] + filter"}
DS = 0.5

plt.rcParams.update({"font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
                     "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
                     "lines.solid_capstyle": "round"})


def load(scenario, N, ctrl, tag):
    ctag = {"proposed": "", "baseline": "_baseline", "se3": "_se3"}[ctrl]
    f = os.path.join(RESULTS, f"{scenario}_N{N}{ctag}{tag}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    names = sorted(k for k in d.files if k.startswith("quad"))
    return {"t": d["t"], "S": {n: d[n] for n in names}, "names": names}


def zdes(x):
    return circ.Z0 + circ.Z_AMP * np.sin(circ.Z_OMEGA * np.asarray(x))


def circ_channels(S, n, t):
    cx, cy = circ.CENTERS[n]
    er = (np.hypot(S[:, 6] - cx, S[:, 7] - cy) - circ.R) * 100.0            # cm
    ez = (S[:, 8] - zdes(S[:, 6])) * 100.0                                   # cm
    ang = np.unwrap(np.arctan2(S[:, 7] - cy, S[:, 6] - cx))
    eta2 = np.gradient(ang, t)                                               # rad/s
    yaw = np.degrees(S[:, 2])
    return er, ez, eta2, yaw


def sine_channels(S, n, t):
    ey = (S[:, 7] - sine.SIGN[n] * sine.A_SINE * np.sin(sine.W_SINE * S[:, 6])) * 100.0
    ez = (S[:, 8] - zdes(S[:, 6])) * 100.0
    eta2 = np.gradient(S[:, 6], t)                                           # x-rate m/s
    yaw = np.degrees(S[:, 2])
    return ey, ez, eta2, yaw


def min_dist(run):
    t = run["t"]; names = run["names"]
    md = np.full(len(t), np.inf)
    for a, b in itertools.combinations(names, 2):
        md = np.minimum(md, np.linalg.norm(run["S"][a][:, 6:9] - run["S"][b][:, 6:9], axis=1))
    return md


def path_xyz(scenario, n, npts=400):
    if scenario == "circles":
        p = circ.make_configs(names=(n,) if n == "quad_A" else ("quad_A", "quad_B", "quad_C", "quad_D"))
        pth = [c.path for c in p if c.name == n][0]
        q = np.linspace(0, 2 * np.pi, npts)
    else:
        pth = sine.make_path(n)
        q = np.linspace(sine.X0_START[n] - 0.5, 10.0, npts)
    return np.array([pth.sig(v, 0) for v in q])


# --------------------------------------------------------------- trajectory figures
def traj_figure(scenario, N, tag, out, layout, top_xlim=None):
    plt.rcParams.update({"font.size": 14, "axes.labelsize": 14, "xtick.labelsize": 12,
                         "ytick.labelsize": 12, "legend.fontsize": 12.5})
    runs = {c: load(scenario, N, c, tag) for c in ("proposed", "baseline", "se3")}
    runs = {c: r for c, r in runs.items() if r is not None}
    names = runs["proposed"]["names"]
    if layout == "lr":
        fig = plt.figure(figsize=(7.0, 2.4))
        ax3 = fig.add_subplot(1, 2, 1, projection="3d")
        ax2 = fig.add_subplot(1, 2, 2)
    else:
        fig = plt.figure(figsize=(3.6, 5.4))
        ax3 = fig.add_subplot(2, 1, 1, projection="3d")
        ax2 = fig.add_subplot(2, 1, 2)
    for n in names:
        G = path_xyz(scenario, n)
        ax3.plot(G[:, 0], G[:, 1], G[:, 2], color="0.55", lw=0.7, ls=(0, (1, 1.2)))
        ax2.plot(G[:, 0], G[:, 1], color="0.55", lw=0.7, ls=(0, (1, 1.2)))
    for ctrl, run in runs.items():
        ls, lw = STYLE[ctrl]
        for n in names:
            S = run["S"][n]
            ax3.plot(S[:, 6], S[:, 7], S[:, 8], ls, color=COLORS[n], lw=lw)
            ax2.plot(S[:, 6], S[:, 7], ls, color=COLORS[n], lw=lw)
            if ctrl == "proposed":
                ax3.scatter(*S[0, 6:9], color=COLORS[n], s=9, marker="o", depthshade=False)
    ax3.set_xlabel("x [m]", labelpad=-4); ax3.set_ylabel("y [m]", labelpad=-4)
    ax3.set_zlabel("z [m]", labelpad=-4)
    ax3.tick_params(pad=-2)
    ax3.view_init(elev=28, azim=-60)
    try:
        ax3.set_box_aspect(None, zoom=1.2)
    except TypeError:
        pass
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    if top_xlim is not None:
        ax2.set_xlim(*top_xlim)
        ax2.set_ylim(-6.2, 6.2)
        ax2.set_aspect("equal", adjustable="box")
    else:
        ax2.set_aspect("equal", adjustable="datalim")
    handles = [Line2D([0], [0], color="k", ls=STYLE[c][0], lw=1.8, label=LABEL[c])
               for c in runs] + [Line2D([0], [0], color="0.55", ls=(0, (1, 1.2)), lw=1.0, label="paths")]
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.09),
               frameon=False, handlelength=1.7, columnspacing=1.2)
    fig.tight_layout(pad=0.4, rect=(0, 0, 1, 0.97))
    fig.subplots_adjust(wspace=0.30)   # keep the 3D z-label clear of the top view's y-label
    fig.savefig(out, format="eps", bbox_inches="tight")
    fig.savefig(out.replace(".eps", ".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


# --------------------------------------------------------------- 2x5 channels figure
def channels_figure(tag, out, which=("circles", "sine")):
    plt.rcParams.update({"font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11.5,
                         "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 11.5})
    all_rows = {"circles": ("circles", 4, circ_channels, "rad/s", 0.5),
                "sine": ("sine", 2, sine_channels, "m/s", 0.8)}
    sel = [all_rows[k] for k in which]
    fig, axes = plt.subplots(len(sel), 5, figsize=(10.5, 1.45 * len(sel) + 0.3), squeeze=False)
    titles = ["transverse error [cm]", "altitude error [cm]", "along-path speed",
              "heading [deg]", "min pairwise distance [m]"]
    for r, (scenario, N, chan, spdunit, ds_row) in enumerate(sel):
        runs = {c: load(scenario, N, c, tag) for c in ("proposed", "baseline", "se3")}
        runs = {c: v for c, v in runs.items() if v is not None}
        names = runs["proposed"]["names"]
        for ctrl, run in runs.items():
            t = run["t"]; ls, lw = STYLE[ctrl]
            for n in names:
                er, ez, e2, yw = chan(run["S"][n], n, t)
                axes[r, 0].plot(t, er, ls, color=COLORS[n], lw=lw)
                axes[r, 1].plot(t, ez, ls, color=COLORS[n], lw=lw)
                axes[r, 2].plot(t, e2, ls, color=COLORS[n], lw=lw)
                axes[r, 3].plot(t, yw, ls, color=COLORS[n], lw=lw)
            axes[r, 4].plot(t, min_dist(run), ls, color="k" if ctrl == "proposed" else "0.5", lw=lw)
        if scenario == "circles":
            for n in names:
                axes[r, 2].axhline(circ.V_DES[n], color=COLORS[n], lw=0.5, alpha=0.4)
        else:
            for n in names:
                axes[r, 2].axhline(sine.V_DES[n], color=COLORS[n], lw=0.5, alpha=0.4)
        axes[r, 4].axhline(ds_row, color="r", lw=0.7, ls="--")
        for cidx in range(5):
            axes[r, cidx].set_xlabel("t [s]", labelpad=1)
            if r == 0 or cidx == 2:
                axes[r, cidx].set_title(titles[cidx] if cidx != 2
                                        else f"along-path speed [{spdunit}]", pad=3)
    handles = [Line2D([0], [0], color="k", ls=STYLE[c][0], lw=STYLE[c][1], label=LABEL[c])
               for c in ("proposed", "baseline", "se3")]
    handles.append(Line2D([0], [0], color="r", ls="--", lw=0.7, label="$d_s$"))
    fig.legend(handles=handles, ncol=4, loc="upper center", bbox_to_anchor=(0.5, 1.17),
               frameon=False, handlelength=1.7, columnspacing=1.4)
    fig.tight_layout(pad=0.45, rect=(0, 0, 1, 0.97))
    fig.savefig(out, format="eps", bbox_inches="tight")
    fig.savefig(out.replace(".eps", ".png"), dpi=250, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_paper")
    ap.add_argument("--outdir", default=os.path.join(RESULTS, "figs"))
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)
    traj_figure("circles", 4, args.tag, os.path.join(args.outdir, "pf_circ_n4.eps"), "lr")
    traj_figure("sine", 2, args.tag, os.path.join(args.outdir, "pf_sin_n2.eps"), "lr", top_xlim=(-9.0, 9.0))
    channels_figure(args.tag, os.path.join(args.outdir, "fig_paper_4var_combined.eps"),
                    which=("circles",))
    channels_figure(args.tag, os.path.join(args.outdir, "fig_channels_web.eps"))


if __name__ == "__main__":
    main()
