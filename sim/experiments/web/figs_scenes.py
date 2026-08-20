r"""Project-page scenario figures: the three-way controller comparison, per channel and in space.

Same data as the manuscript figures, redrawn at web size with type large enough to read in a browser.
"""
import argparse, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from mpl_toolkits.mplot3d import Axes3D  # noqa

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "tflqp"))
import style as S
import webqc
import paths as P

RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "web")
SPACER = 0.20      # width of the empty column that holds the 3d z axis furniture

CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}
CTRL = [("", "-", 2.0, "proposed", S.CTRL["proposed"]),
        ("_baseline", "--", 1.5, "TFL + safety filter", S.CTRL["baseline"]),
        ("_se3", ":", 1.7, "geometric + safety filter", S.CTRL["se3"])]


def get_path(scn, n, dg):
    if scn == "sine":
        import scenario_sine as ss
        w = float(dg["sine_w"]) if "sine_w" in dg.files else ss.W_SINE
        A = float(dg["sine_A"]) if "sine_A" in dg.files else ss.A_SINE
        return P.mirrored_sine(n, sign=ss.SIGN[n], A=A, w=w, z_amp=ss.Z_AMP, z_omega=ss.Z_OMEGA,
                               z0=ss.Z0, x_shift=0.0)
    return P.lifted_circle(n, *CEN[n], R=1.5)


def load_runs(scn, N, tag):
    dg = np.load(os.path.join(RES, f"{scn}_N{N}{tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in dg["names"]]
    runs = []
    for sfx, ls, lw, lab, col in CTRL:
        f = os.path.join(RES, f"{scn}_N{N}{sfx}{tag}.npz")
        if os.path.exists(f):
            Sd = np.load(f)
            runs.append(({n: Sd[n] for n in names}, Sd["t"], ls, lw, lab, col))
    return dg, names, runs


def transverse(scn, n, X, dg):
    pth = get_path(scn, n, dg)
    sv = np.array([pth.s_val(y) for y in X[:, 6:9]])
    return sv[:, 0] * 100.0, sv[:, 1] * 100.0


def minbar(st, names, ds):
    m = None
    for a, b in itertools.combinations(names, 2):
        d = st[a][:, 6:9] - st[b][:, 6:9]
        v = np.einsum("ij,ij->i", d, d) - ds ** 2
        m = v if m is None else np.minimum(m, v)
    return m


def fig_channels(scn, N, tag, out, title, display_px=1180):
    dg, names, runs = load_runs(scn, N, tag)
    ds = float(dg["ds"])
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))
    fig.subplots_adjust(wspace=0.32, left=0.062, right=0.988, top=0.80, bottom=0.30)

    nmin = min(len(t) for _, t, _, _, _, _ in runs)
    t = runs[0][1][:nmin]
    for st, _, ls, lw, lab, col in runs:
        # |xi_1| alone, which is the transverse output the manuscript reports and the table quotes
        e = np.zeros(nmin)
        for n in names:
            xi, ze = transverse(scn, n, st[n][:nmin], dg)
            e = np.maximum(e, np.abs(xi))
        ax[0].semilogy(t, np.maximum(e, 1e-3), ls=ls, lw=lw, color=col, label=lab)
        yaw = np.zeros(nmin)
        for n in names:
            yaw = np.maximum(yaw, np.abs(np.degrees(st[n][:nmin, 2])))
        ax[1].semilogy(t, np.maximum(yaw, 1e-6), ls=ls, lw=lw, color=col, label=lab)
        ax[2].plot(t, minbar({n: st[n][:nmin] for n in names}, names, ds), ls=ls, lw=lw, color=col,
                   label=lab)

    ax[0].set_ylabel(r"worst $|\xi_1^i|\times 100$")
    S.panel_title(ax[0], "(a)  transverse output")
    ax[1].set_ylabel("worst heading error [deg]")
    S.panel_title(ax[1], "(b)  how far from the assigned heading")
    ax[2].axhline(0, color=S.DANGER, lw=1.5, label="contact, $h_{ij}=0$")
    ax[2].set_ylabel(r"$\min_{ij}\,h_{ij}$  [m$^2$]")
    S.panel_title(ax[2], "(c)  collision barrier")
    for a in ax:
        a.set_xlim(t[0], t[-1]); a.set_xlabel("time [s]")
        a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    ax[2].set_ylim(-0.35, None)
    webqc.make_room(ax[2])

    handles = [Line2D([], [], color=c, lw=lw, ls=ls, label=lab) for _, ls, lw, lab, c in CTRL]
    handles.append(Line2D([], [], color=S.DANGER, lw=1.5, label="contact, $h_{ij}=0$"))
    fig.legend(handles=handles, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.015),
               fontsize=12, handlelength=2.6, columnspacing=1.8, frameon=False)
    fig.suptitle(title, x=0.062, ha="left", fontsize=16, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


def fig_space(scn, N, tag, out, title, display_px=1180):
    dg, names, runs = load_runs(scn, N, tag)
    S.use(1.0)
    fig = plt.figure(figsize=(13.4, 5.0))
    # mplot3d hangs the z axis furniture off the right of its box, straight into the panel beside
    # it. An empty middle column reserves that space instead of hoping wspace is enough.
    gs = fig.add_gridspec(1, 3, width_ratios=[1.14, SPACER, 1.0], wspace=0.05,
                          left=0.005, right=0.985, top=0.815, bottom=0.10)
    a3 = fig.add_subplot(gs[0], projection="3d")
    at = fig.add_subplot(gs[2])
    # 30 m downrange in a 4 m corridor, so crop to the stretch holding the two crossings
    xwin = (1.0, 16.0) if scn == "sine" else None
    a3.set_box_aspect((2.0, 1, 0.62) if scn == "sine" else (1, 1, 0.5), zoom=1.34)

    for n in names:
        pth = get_path(scn, n, dg)
        if scn == "sine":
            qq = np.linspace(*xwin, 600)
        else:
            qq = np.linspace(0, 2 * np.pi, 400)
        c = np.array([pth.sig(q, 0) for q in qq])
        # a wide pale ribbon, not a hairline: the proposed trace lies on top of it
        a3.plot(c[:, 0], c[:, 1], c[:, 2], color="#C4CED7", lw=4.2, solid_capstyle="round", zorder=1)
        at.plot(c[:, 0], c[:, 1], color="#C4CED7", lw=4.2, solid_capstyle="round", zorder=1)

    for st, _, ls, lw, lab, col in runs:
        for n in names:
            X = st[n]
            if xwin is not None:
                k = (X[:, 6] >= xwin[0]) & (X[:, 6] <= xwin[1])
                X = X[k]
            a3.plot(X[:, 6], X[:, 7], X[:, 8], color=S.AGENT[n], lw=lw, ls=ls, zorder=3)
            at.plot(X[:, 6], X[:, 7], color=S.AGENT[n], lw=lw, ls=ls, zorder=3)
    for n in names:
        X = runs[0][0][n]
        a3.scatter(X[0, 6], X[0, 7], X[0, 8], color=S.AGENT[n], s=42, depthshade=False, zorder=5,
                   edgecolor="k", linewidth=0.6)
        at.scatter(X[0, 6], X[0, 7], color=S.AGENT[n], s=42, zorder=5, edgecolor="k", linewidth=0.6)

    a3.set_xlabel("$x$ [m]", labelpad=2); a3.set_ylabel("$y$ [m]", labelpad=2)
    a3.set_zlabel("$z$ [m]", labelpad=0); a3.tick_params(labelsize=11, pad=1)
    a3.set_title("(a)  three-dimensional view", loc="left", fontweight="semibold", pad=-10, x=0.10)
    if xwin is not None:
        a3.set_xlim(*xwin); at.set_xlim(*xwin)
    at.set_aspect("equal", adjustable="box")
    at.set_xlabel("$x$ [m]"); at.set_ylabel("$y$ [m]")
    at.set_title("(b)  seen from above", loc="left", fontweight="semibold", pad=8)

    handles = [Line2D([], [], color="#3A4550", lw=lw, ls=ls, label=lab) for _, ls, lw, lab, _ in CTRL]
    handles.append(Line2D([], [], color="#C4CED7", lw=4.2, label="assigned path"))
    handles.append(Line2D([], [], color="#3A4550", marker="o", ls="none", ms=6, label="start"))
    fig.legend(handles=handles, ncol=5, loc="upper center", bbox_to_anchor=(0.5, 0.955), fontsize=12,
               handlelength=2.6, columnspacing=1.6, frameon=False)
    fig.suptitle(title, x=0.03, ha="left", fontsize=16, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px, require_labels=False)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all")
    ap.add_argument("--spacer", type=float, default=SPACER)
    a = ap.parse_args()
    SPACER = a.spacer
    os.makedirs(OUT, exist_ok=True)
    if a.which in ("all", "channels"):
        fig_channels("circles", 4, "_cs1p4", "W_channels_circles",
                     "Four quadrotors on intersecting circles: proposed against two cascades")
        fig_channels("sine", 2, "", "W_channels_sine",
                     "Two quadrotors on intersecting sinusoids: proposed against two cascades")
    if a.which in ("all", "space"):
        fig_space("circles", 4, "_cs1p4", "W_space_circles",
                  "Four quadrotors on intersecting circles")
        fig_space("sine", 2, "", "W_space_sine", "Two quadrotors on intersecting sinusoids")
