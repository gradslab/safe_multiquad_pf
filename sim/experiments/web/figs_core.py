r"""Project-page figures built from the per-step diagnostics of one rollout.

Slack interval and row authorities through an encounter, and path distance against the decoupling
matrix conditioning. Every figure is gated by webqc before it is written.
"""
import argparse, itertools, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "tflqp"))
import style as S
import webqc
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "web")


def load(tag):
    d = np.load(os.path.join(RES, f"{tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    return d, names, d["t"]


def encounter_window(d, names, pad=1.4):
    """Time window around the closest approach, which is where every constraint does its work."""
    t = d["t"]
    best = None
    for a, b in itertools.combinations(names, 2):
        k = f"dist_{a}_{b}"
        if k not in d.files:
            continue
        dd = d[k]
        i = int(np.nanargmin(np.where(t > 0.8, dd, np.inf)))
        if best is None or dd[i] < best[1]:
            best = (i, dd[i], (a, b))
    if best is None:
        return t[0], t[-1], None
    i = best[0]
    return max(t[0], t[i] - pad), min(t[-1], t[i] + pad), best


# ---------------------------------------------------------------- interval and authority

def _corner(ax, nx=5):
    """Drop the x tick sitting in the bottom-left corner, where it collides with the y ticks."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=nx, prune="lower"))


def _headroom(ax, frac=0.34, bottom=0.06):
    """Reserve space above the data so a legend never sits on a curve."""
    lo, hi = ax.get_ylim(); r = hi - lo
    ax.set_ylim(lo - bottom * r, hi + frac * r)


def fig_authority(tag, out, title_scn, display_px=1180):
    d, names, t = load(tag)
    ds = float(d["ds"])
    t0, t1, best = encounter_window(d, names, pad=2.0)
    S.use(1.0)
    fig, ax = plt.subplots(2, 2, figsize=(13.4, 8.8))
    fig.subplots_adjust(hspace=0.55, wspace=0.34, left=0.085, right=0.985, top=0.905, bottom=0.095)

    # (a) pairwise distance against the separation requirement
    a0 = ax[0, 0]
    pairs = [(a, b) for a, b in itertools.combinations(names, 2) if f"dist_{a}_{b}" in d.files]
    # pairs get their own ramp; the agent colours would read as agent identity
    pcol = plt.cm.cividis(np.linspace(0.05, 0.85, max(len(pairs), 1)))
    for c, (a, b) in zip(pcol, pairs):
        a0.plot(t, d[f"dist_{a}_{b}"], lw=1.6, color=c, label=f"{a[-1]}\u2013{b[-1]}")
    a0.axhline(ds, color=S.DANGER, lw=1.6, ls="--", label=f"separation $d_s = {ds:g}$ m")
    a0.set_xlim(t[0], t[-1]); a0.set_ylim(0, None)
    _headroom(a0, 0.40, 0.0)
    a0.set_xlabel("time [s]"); a0.set_ylabel("distance [m]")
    S.panel_title(a0, "(a)  every pair stays outside the separation sphere")
    a0.legend(ncol=3, loc="upper center", fontsize=10.5); webqc.make_room(a0)

    # (b) the slack the projection selects, against the bound that actually binds
    a1 = ax[0, 1]
    n = names[0]
    lo, hi, dstar = d[f"{n}_delta_lo"], d[f"{n}_delta_hi"], d[f"{n}_delta_star"]
    m = (t >= t0) & (t <= t1)
    a1.fill_between(t[m], np.nanmin(dstar[m]) - 50, hi[m], color="#DCE9F4", zorder=0,
                    label="feasible side of the binding bound")
    a1.plot(t[m], hi[m], color="#5B8DB8", lw=1.8, label=r"upper bound $\delta^{+}$")
    a1.plot(t[m], dstar[m], color=S.CTRL["proposed"], lw=2.2, label=r"applied $\delta^{\star}$")
    a1.set_xlim(t0, t1)
    bind = m & (np.abs(dstar - hi) < 1e-9)
    ylo = min(np.nanmin(dstar[m]), np.nanmin(hi[m])); yhi = max(np.nanmax(dstar[m]), np.nanmax(hi[m]))
    r = max(yhi - ylo, 1e-6)
    a1.set_ylim(ylo - 0.20 * r, yhi + 0.52 * r)
    yb = ylo - 0.145 * r
    # drawn as a masked series, not as t[bind]: plotting only the selected samples would join the
    # gaps and claim the row binds where it does not
    bar = np.where(bind, yb, np.nan)
    a1.plot(t[m], bar[m], color=S.DANGER, lw=5.0, solid_capstyle="butt",
            label="collision row binding")
    a1.set_xlabel("time [s]"); a1.set_ylabel(r"slack $\delta$  [m/s$^4$]")
    S.panel_title(a1, f"(b)  agent {n[-1]}: the slack rides the collision bound")
    a1.legend(loc="upper center", fontsize=10.5); webqc.make_room(a1)
    print(f"      note: {out} lower bound stays below {np.nanmax(lo[m]):.0f} m/s^4 through the window")

    # (c) interval width, every agent, over the whole run
    a2 = ax[1, 0]
    for n2 in names:
        a2.semilogy(t, d[f"{n2}_width"], lw=1.5, color=S.AGENT.get(n2, S.INK), label=n2[-1])
    a2.set_xlim(t[0], t[-1])
    wlo = min(np.nanmin(d[f"{n2}_width"]) for n2 in names)
    whi = max(np.nanmax(d[f"{n2}_width"]) for n2 in names)
    a2.set_ylim(wlo * 0.55, whi * 6.0)
    a2.set_xlabel("time [s]"); a2.set_ylabel(r"interval width $\delta^{+}-\delta^{-}$  [m/s$^4$]")
    S.panel_title(a2, "(c)  authority left to the safety filter, never zero on this run")
    a2.legend(ncol=len(names), loc="upper center", fontsize=11, title="agent", title_fontsize=11)
    webqc.make_room(a2)

    # (d) row authorities: which constraint can move the slack, and by how much
    a3 = ax[1, 1]
    n = names[0]
    others = [x for x in names if x != n]
    ac = d[f"{n}_abar_coll"]
    for j in range(ac.shape[1]):
        a3.plot(t[m], ac[m, j], lw=1.8, color=S.AGENT[others[j]],
                label=rf"collision with {others[j][-1]}")
    asp = d[f"{n}_abar_speed"]
    a3.plot(t[m], asp[m, 0], lw=1.5, ls="--", color=S.MUTED, label=r"speed rows, $\pm1$")
    a3.plot(t[m], asp[m, 1], lw=1.5, ls="--", color=S.MUTED)
    a3.axhline(0, color="#8A9199", lw=1.2, zorder=1)
    a3.set_xlim(t0, t1)
    amax = np.nanmax(np.abs(ac[m])) * 1.12
    a3.set_ylim(-amax, amax * 1.62)
    a3.set_xlabel("time [s]"); a3.set_ylabel(r"row authority $\bar a_k$")
    S.panel_title(a3, f"(d)  agent {n[-1]}: collision authority crosses zero once per encounter")
    a3.legend(loc="upper center", fontsize=10.5, ncol=2); webqc.make_room(a3)

    for a in ax.ravel():
        _corner(a)
    fig.suptitle(title_scn, x=0.085, ha="left", fontsize=16, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


# ---------------------------------------------------------------- path distance and conditioning

def fig_pathdist(tag, out, title_scn, display_px=1180):
    d, names, t = load(tag)
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.1))
    fig.subplots_adjust(wspace=0.34, left=0.062, right=0.988, top=0.855, bottom=0.165)

    a0 = ax[0]
    for n in names:
        a0.semilogy(t, np.maximum(d[f"{n}_path_err"] * 100, 1e-4), lw=1.5,
                    color=S.AGENT.get(n, S.INK), label=n[-1])
    a0.set_xlim(t[0], t[-1]); a0.set_xlabel("time [s]")
    a0.set_ylabel(r"$\mathrm{dist}(x_q^i,\gamma^i)$  [cm]")
    S.panel_title(a0, "(a)  distance to the path")
    a0.legend(ncol=len(names), loc="upper right", fontsize=11, title="agent", title_fontsize=11)
    webqc.make_room(a0)

    a1 = ax[1]
    for n in names:
        xi = d[f"{n}_xi"][:, 0]; ze = d[f"{n}_zeta"][:, 0]
        a1.semilogy(t, np.maximum(np.hypot(xi, ze), 1e-9), lw=1.5, color=S.AGENT.get(n, S.INK),
                    label=n[-1])
    a1.set_xlim(t[0], t[-1]); a1.set_xlabel("time [s]")
    a1.set_ylabel(r"$\|(\xi_1^i,\zeta_1^i)\|$")
    S.panel_title(a1, "(b)  transverse output")


    a2 = ax[2]
    for n in names:
        a2.plot(t, d[f"{n}_sigma_min_D"], lw=1.5, color=S.AGENT.get(n, S.INK), label=n[-1])
    lo = min(np.nanmin(d[f"{n}_sigma_min_D"]) for n in names)
    a2.axhline(lo, color=S.MUTED, lw=1.2, ls="--", label=f"run minimum, {lo:.2f}")
    a2.set_xlim(t[0], t[-1]); a2.set_ylim(0, None)
    a2.set_xlabel("time [s]"); a2.set_ylabel(r"$\sigma_{\min}(D^i)$")
    S.panel_title(a2, "(c)  conditioning of $D^i$")
    a2.legend(loc="lower right", fontsize=11)
    webqc.make_room(a2)

    fig.suptitle(title_scn, x=0.062, ha="left", fontsize=16, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", default="all")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.which in ("all", "authority"):
        fig_authority("circles_N4_cs1p4", "W_authority_circles",
                      "Four quadrotors, intersecting circles")
        fig_authority("sine_N2", "W_authority_sine", "Two quadrotors, intersecting sinusoids")
    if a.which in ("all", "pathdist"):
        fig_pathdist("circles_N4_cs1p4", "W_pathdist_circles",
                     "Four quadrotors, intersecting circles")
        fig_pathdist("sine_N2", "W_pathdist_sine", "Two quadrotors, intersecting sinusoids")
