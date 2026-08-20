r"""Single condensed figure for the whole simulation suite, in the manuscript's combined format.

One double-column figure*: rows are scenarios, columns are the output channels that the theorems
speak about. Proposed solid, SE(3) tracker + CBF filter dashed, per-agent colour. No suptitle; the
caption carries the message.

    (a) transverse error xi_1     (b) altitude error zeta_1   (c) along-path speed eta_2
    (d) heading mu_1              (e) collision barrier min_ij h_ij

Usage: python3 experiments/fig_suite.py
"""
import itertools, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp")); sys.path.insert(0, PKG)
import paths as P
import qc_figures
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)
# Also write straight into the paper's figures/ directory. Copying by hand meant the paper embedded a
# stale figure whenever a fix landed after the copy; writing both places removes that failure mode.
PAPER_FIG = os.path.abspath(os.path.join(HERE, "..", "..", "figures"))

COL = {"quad_A": "#4477AA", "quad_B": "#EE6677", "quad_C": "#228833", "quad_D": "#CCBB44"}
CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}
DS = 0.5      # overridden per scenario from the diag npz
ROWS = [("circles", 4, "N = 4 circles"), ("sine", 2, "N = 2 sine")]
OUT = "F_suite_combined"
CTRL = [("", "-", "proposed"),
        ("_baseline", "--", "TFL + safety filter"),
        ("_se3", ":", "[31] + safety filter")]
TAGS = {"circles": "", "sine": ""}   # per-scenario parameter variant, set from the CLI
PAPER = False


GEO = {}     # geometry the data was actually generated with, read from the diag npz


def get_path(sc, n):
    """Resolve the path the RUN used. The sine geometry is a run-time parameter, so it must come from
    the saved diagnostics; reading module defaults here silently measured the error against the wrong
    path once already."""
    if sc == "sine":
        import scenario_sine as s
        w = GEO.get("sine_w", s.W_SINE); A = GEO.get("sine_A", s.A_SINE)
        return P.mirrored_sine(n, sign=s.SIGN[n], A=A, w=w,
                               z_amp=s.Z_AMP, z_omega=s.Z_OMEGA, z0=s.Z0, x_shift=0.0)
    return P.lifted_circle(n, *CEN[n], R=1.5)


def transverse(sc, n, S):
    """xi_1 = s_1(x_q) and zeta_1 = s_2(x_q), in cm, straight from the path's level sets."""
    pth = get_path(sc, n)
    sv = np.array([pth.s_val(y) for y in S[:, 6:9]])
    return sv[:, 0] * 100.0, sv[:, 1] * 100.0


def tangential(sc, n, S, qw=0.0):
    pth = get_path(sc, n); out = np.empty(len(S)); q = qw
    for k, y in enumerate(S[:, 6:9]):
        q = pth.q_star(y, q)
        tv = pth.sig(q, 1); tv = tv / max(np.linalg.norm(tv), 1e-9)
        out[k] = tv @ S[k, 9:12]
    return out


def minbar(states, names):
    m = None
    for a, b in itertools.combinations(names, 2):
        d = states[a][:, 6:9] - states[b][:, 6:9]
        v = np.einsum("ij,ij->i", d, d) - DS ** 2
        m = v if m is None else np.minimum(m, v)
    return m


def panel(ax, t, series, ylab, deslines=None, zero=True):
    """series: list of (values, colour, linestyle)."""
    vals = []
    w = t > 0.4
    for y, c, ls in series:
        ax.plot(t, y, color=c, lw=1.25 if ls == "-" else 1.1, ls=ls, alpha=1.0 if ls == "-" else 0.9)
        vals.append(np.asarray(y)[w])
    if deslines:
        for v, c in deslines:
            ax.axhline(v, color=c, lw=0.5, ls=":", alpha=0.55)
    lo = min(np.nanmin(v) for v in vals); hi = max(np.nanmax(v) for v in vals)
    pad = 0.15 * (hi - lo) + 1e-6
    ax.set_ylim(lo - pad, hi + pad); ax.set_xlim(0, t[-1])
    if zero:
        ax.axhline(0, color="0.6", lw=0.6)
    ax.set_ylabel(ylab, fontsize=7 if PAPER else 10)
    ax.tick_params(labelsize=6.5 if PAPER else 8)
    if PAPER:
        from matplotlib.ticker import MaxNLocator
        ax.xaxis.set_major_locator(MaxNLocator(nbins=4, prune="lower"))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4))


def main():
    # PAPER: authored at the width it is printed at, so 7 pt on this canvas is 7 pt on the page.
    # The old 16 in canvas was scaled to 6.3 in by LaTeX, which took every label down to about 4 pt.
    if PAPER:
        plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7.5,
                             "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 7,
                             "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                             "ytick.major.width": 0.6, "lines.linewidth": 0.9})
        fig, ax = plt.subplots(len(ROWS), 5, figsize=(7.0, 1.95 if len(ROWS) == 1 else 2.2 * len(ROWS)),
                               squeeze=False)
        fig.subplots_adjust(wspace=0.62, hspace=0.40, top=0.88,
                            bottom=0.30 if len(ROWS) == 1 else 0.16, left=0.058, right=0.995)
    else:
        fig, ax = plt.subplots(len(ROWS), 5,
                               figsize=(16.2, 2.45 if len(ROWS) == 1 else 2.7 * len(ROWS)),
                               squeeze=False)
        fig.subplots_adjust(wspace=0.58, hspace=0.34, top=0.93,
                            bottom=0.30 if len(ROWS) == 1 else 0.15, left=0.040, right=0.998)

    for r, (sc, N, label) in enumerate(ROWS):
        base = os.path.join(RES, f"{sc}_N{N}{TAGS.get(sc, '')}")
        if not os.path.exists(base + ".npz"):
            for j in range(5):
                ax[r, j].text(.5, .5, f"{sc} N={N}\nnot run", ha="center", va="center",
                              transform=ax[r, j].transAxes, fontsize=9, color="0.5")
                ax[r, j].set_xticks([]); ax[r, j].set_yticks([])
            continue
        dg = np.load(os.path.join(RES, f"{sc}_N{N}{TAGS.get(sc, '')}_diag.npz"), allow_pickle=True)
        GEO.clear()
        for k in ("sine_w", "sine_A"):
            if k in dg.files:
                GEO[k] = float(dg[k])
        if sc == "sine" and not GEO:
            print(f"  WARNING: {sc} data has no geometry metadata; falling back to module defaults")
        names = [str(x) for x in dg["names"]]
        runs = []                                    # (states, linestyle)
        tmin = None
        for sfx, ls, _lab in CTRL:
            f = os.path.join(RES, f"{sc}_N{N}{sfx}{TAGS.get(sc, '')}.npz")
            if not os.path.exists(f):
                continue
            S = np.load(f)
            runs.append(({n: S[n] for n in names}, ls, S["t"]))
            tmin = len(S["t"]) if tmin is None else min(tmin, len(S["t"]))
        if not runs:
            continue
        t = runs[0][2][:tmin]
        runs = [({n: st[n][:tmin] for n in names}, ls, None) for st, ls, _ in runs]
        vdes = {n: float(dg["v_des"][i]) for i, n in enumerate(names)}

        chans = []
        for st, ls, _ in runs:
            xi, ze, et, yw = {}, {}, {}, {}
            for n in names:
                xi[n], ze[n] = transverse(sc, n, st[n])
                et[n] = tangential(sc, n, st[n]); yw[n] = np.degrees(st[n][:, 2])
            chans.append((xi, ze, et, yw, st, ls))

        def ser(idx):
            out = []
            for c in chans:
                d, ls = c[idx], c[5]
                out += [(d[n], COL[n], ls) for n in names]
            return out

        panel(ax[r, 0], t, ser(0), r"$\xi_1^i$ [cm]")
        panel(ax[r, 1], t, ser(1), r"$\zeta_1^i$ [cm]")
        panel(ax[r, 2], t, ser(2), r"$\eta_2^i$ [m/s]",
              deslines=[(vdes[n], COL[n]) for n in names])
        panel(ax[r, 3], t, ser(3), r"$\mu_1^i$ [deg]")
        global DS
        DS = float(dg["ds"])
        s = [(minbar(c[4], names), "#333333", c[5]) for c in chans]
        panel(ax[r, 4], t, s, r"$\min_{ij} h_{ij}$ [m$^2$]", zero=False)
        ax[r, 4].axhline(0, color="crimson", lw=0.9)
        if len(ROWS) > 1:
            ax[r, 0].annotate(label, xy=(-0.40, 0.5), xycoords="axes fraction", rotation=90,
                              va="center", ha="center", fontsize=7 if PAPER else 10,
                              fontweight="bold")

    for j, ttl in enumerate(["(a) transverse error", "(b) altitude error", "(c) along-path speed",
                             "(d) heading", "(e) collision barrier"]):
        ax[0, j].set_title(ttl, fontsize=7.5 if PAPER else 11)
        ax[-1, j].set_xlabel(r"$t$ [s]", fontsize=7 if PAPER else 10)
    fig.legend([Line2D([0], [0], color="0.3", lw=1.5 if PAPER else 2.2, ls=ls) for _, ls, _ in CTRL],
               [lab for _, _, lab in CTRL], ncol=3, loc="lower center",
               bbox_to_anchor=(0.5, (-0.04 if PAPER else -0.03) if len(ROWS) > 1 else
                               (-0.02 if PAPER else -0.015)),
               fontsize=7 if PAPER else 11, frameon=False,
               handlelength=2.2 if PAPER else 2.6, columnspacing=1.2 if PAPER else 1.6)
    if PAPER:
        sys.path.insert(0, os.path.join(HERE, "web"))
        import webqc
        issues = webqc.gate_print(fig, OUT, printed_in=7.0, min_pt=6.0, require_labels=False)
    else:
        issues = qc_figures.check(fig, OUT)
    for e in (".pdf", ".png", ".eps"):
        fig.savefig(os.path.join(FIG, OUT + e), bbox_inches="tight", pad_inches=0.01,
                    **({"dpi": 200} if e == ".png" else {}))
    if os.path.isdir(PAPER_FIG):
        fig.savefig(os.path.join(PAPER_FIG, OUT + ".pdf"), bbox_inches="tight", pad_inches=0.01)
    print(f"wrote {OUT}.pdf/.png/.eps")
    return 1 if issues else 0


if __name__ == "__main__":
    import argparse
    _ap = argparse.ArgumentParser()
    _ap.add_argument("--tag-circles", default=""); _ap.add_argument("--tag-sine", default="")
    _ap.add_argument("--only", default="", help="restrict to one scenario, e.g. circles")
    _ap.add_argument("--out", default="F_suite_combined")
    _ap.add_argument("--paper", action="store_true", help="author at the printed width, 7 pt type")
    _a = _ap.parse_args()
    TAGS = {"circles": _a.tag_circles, "sine": _a.tag_sine}
    if _a.only:
        ROWS = [r for r in ROWS if r[0] == _a.only]
    OUT = _a.out
    PAPER = _a.paper
    sys.exit(main())
