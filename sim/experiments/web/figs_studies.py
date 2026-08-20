r"""Project-page study figures: feasibility map, failure mechanism, sampling rate, solver cost,
responsibility split, actuator bounds.
"""
import argparse, glob, itertools, json, os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "src", "tflqp"))
import style as S
import webqc
RES = os.path.join(ROOT, "results")
OUT = os.path.join(ROOT, "results", "web")


def grid(kind):
    key = "ds" if kind == "sine" else "center_scale"
    rows = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "sweeps", "symmetric",
                                                               f"{kind}_a*.json"))]
    # the circle grid is stored relative to the reported speeds, the sine grid to its own nominal
    vk = "v_rel" if all("v_rel" in r for r in rows) else "vscale"
    ys = sorted({r[key] for r in rows}); xs = sorted({r[vk] for r in rows})
    surv = np.full((len(ys), len(xs)), np.nan)
    stop = np.full((len(ys), len(xs)), np.nan)
    for r in rows:
        i, j = ys.index(r[key]), xs.index(r[vk])
        surv[i, j] = 0.0 if r["infeasible"] else 1.0
        stop[i, j] = (r["t_infeasible"] / r["tmax"]) if r["infeasible"] else np.nan
    return key, np.array(ys), np.array(xs), surv, stop


def _heat(ax, ys, xs, surv, stop, ylab, xlab, title, mark=None):
    """Survived cells in one colour, stopped cells shaded by how far into the run they got."""
    ny, nx = surv.shape
    for i in range(ny):
        for j in range(nx):
            if np.isnan(surv[i, j]):
                col, txt, tc = "#EDF1F4", "not run", S.MUTED
            elif surv[i, j] > 0.5:
                col, txt, tc = "#BFE3C7", "ran", "#14532D"
            else:
                f = float(np.clip(stop[i, j], 0.05, 1.0))
                col = plt.cm.OrRd(0.78 - 0.42 * f)
                txt, tc = f"{stop[i, j]*100:.0f}%", "#4A0D0D"
            ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor=col, edgecolor=S.PLATE, lw=2.2))
            ax.text(j, i, txt, ha="center", va="center", fontsize=11.5, color=tc, fontweight="medium")
    if mark is not None and mark[0] in list(ys) and mark[1] in list(xs):
        i, j = list(ys).index(mark[0]), list(xs).index(mark[1])
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, facecolor="none", edgecolor=S.INK,
                               lw=2.4, zorder=5))
    ax.set_xlim(-0.5, nx - 0.5); ax.set_ylim(-0.5, ny - 0.5)
    ax.set_xticks(range(nx)); ax.set_xticklabels([f"{v:g}" for v in xs])
    ax.set_yticks(range(ny)); ax.set_yticklabels([f"{v:g}" for v in ys])
    ax.set_xlabel(xlab); ax.set_ylabel(ylab)
    ax.grid(False)
    S.panel_title(ax, title)
    ax._qc_shared_x = False


def fig_heatmaps(display_px=1180):
    S.use(1.0)
    fig, ax = plt.subplots(1, 2, figsize=(13.4, 5.2))
    fig.subplots_adjust(wspace=0.26, left=0.075, right=0.985, top=0.815, bottom=0.275)

    key, ys, xs, surv, stop = grid("sine")
    _heat(ax[0], ys, xs, surv, stop, "separation distance $d_s$ [m]",
          "desired speeds, multiple of nominal", "(a)  two quadrotors, intersecting sinusoids")
    key, ys, xs, surv, stop = grid("circles")
    _heat(ax[1], ys, xs, surv, stop, "path-centre spacing, multiple of nominal",
          "desired speeds, multiple of the reported run",
          "(b)  four quadrotors, intersecting circles", mark=(1.0, 1.0))

    h = [Rectangle((0, 0), 1, 1, facecolor="#BFE3C7", edgecolor="none"),
         Rectangle((0, 0), 1, 1, facecolor=plt.cm.OrRd(0.55), edgecolor="none"),
         Rectangle((0, 0), 1, 1, facecolor="#EDF1F4", edgecolor="none"),
         Rectangle((0, 0), 1, 1, facecolor="none", edgecolor=S.INK, lw=2.0)]
    fig.legend(h, ["ran the full horizon",
                   "certificate reported an empty interval; the label is how far into the horizon",
                   "not run", "the run the paper reports"],
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.025), fontsize=11, frameon=False,
               handlelength=1.5, columnspacing=1.5)
    fig.suptitle("Where the feasibility condition holds", x=0.075, ha="left", fontsize=15,
                 fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, "W_feasibility_map", display_px=display_px)
    plt.close(fig)


def pick_stopped_cell():
    """A cell from the grid that lost feasibility, chosen for how cleanly it shows the mechanism:
    the collision authority nearest zero at the last logged step, with enough history to plot."""
    best = None
    for f in glob.glob(os.path.join(RES, "sweeps", "symmetric", "*_a*.json")):
        r = json.load(open(f))
        if not r["infeasible"]:
            continue
        kind = os.path.basename(f).split("_")[0]
        base = "sine_N2" if kind == "sine" else "circles_N4"
        ax = r["ds"] if kind == "sine" else r["center_scale"]
        tag = f"_sw{kind}_{ax}_{r.get('v_rel', r['vscale'])}".replace(".", "p")
        dp = os.path.join(RES, f"{base}{tag}_diag.npz")
        if not os.path.exists(dp):
            continue
        d = np.load(dp, allow_pickle=True)
        n = str(d["names"][0])
        if d["t"][-1] < 4.0 or f"{n}_abar_coll" not in d.files:
            continue
        a = np.abs(d[f"{n}_abar_coll"][-1]).min()
        if best is None or a < best[0]:
            best = (a, f"{base}{tag}")
    return best[1] if best else None


def fig_mechanism(tag=None, out="W_mechanism", display_px=1180):
    """Why the boundary is a cliff and not a slope: a row authority passes through zero."""
    tag = tag or pick_stopped_cell()
    print(f"      mechanism cell: {tag}")
    d = np.load(os.path.join(RES, f"{tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    t = d["t"]
    n = names[0]
    ac_all = d[f"{n}_abar_coll"]
    j = int(np.argmin(np.abs(ac_all[-1])))       # the neighbour whose row actually ran out
    ac = ac_all[:, j]
    hi, lo, dstar = d[f"{n}_delta_hi"], d[f"{n}_delta_lo"], d[f"{n}_delta_star"]
    t1 = t[-1]; t0 = t1 - 3.0
    m = (t >= t0)
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))
    fig.subplots_adjust(wspace=0.34, left=0.062, right=0.988, top=0.80, bottom=0.20)

    ax[0].plot(t[m], ac[m], lw=2.2, color=S.CTRL["proposed"])
    ax[0].axhline(0, color=S.DANGER, lw=1.4)
    ax[0].set_ylabel(r"collision authority $\bar a_{ij}$")
    S.panel_title(ax[0], "(a)  the authority approaches zero")

    ax[1].plot(t[m], hi[m], lw=2.2, color="#5B8DB8", label=r"upper bound $\delta^{+}$")
    ax[1].plot(t[m], dstar[m], lw=2.0, color=S.CTRL["proposed"], label=r"applied $\delta^{\star}$")
    ax[1].set_ylabel(r"slack $\delta$  [m/s$^4$]")
    S.panel_title(ax[1], "(b)  the bound it produces walks away")
    ax[1].legend(loc="lower left", fontsize=11.5)
    webqc.make_room(ax[1])

    ax[2].plot(t[m], hi[m] - lo[m], lw=2.2, color=S.CTRL["proposed"])
    ax[2].set_ylim(0, None)
    ax[2].set_ylabel(r"interval width  [m/s$^4$]")
    S.panel_title(ax[2], "(c)  width gives no warning: it is wide, then gone")
    for a in ax:
        a.set_xlim(t0, t1)
        a.set_xlabel("time [s]")
        a.axvline(t1, color=S.DANGER, lw=1.6, ls="--")
        a.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    fig.legend([Line2D([], [], color=S.DANGER, lw=1.6, ls="--")],
               ["last feasible step; the next one has an empty interval"],
               ncol=1, loc="lower center", bbox_to_anchor=(0.5, -0.015), fontsize=11.5, frameon=False)
    fig.suptitle("How feasibility is lost, on one cell of the map above",
                 x=0.062, ha="left", fontsize=15, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


# ---------------------------------------------------------------- sampling rate

def fig_rates(out="W_rates", display_px=1180):
    """The controller holds its input between updates; this is what shrinking that period does."""
    rates = [100, 200, 500, 1000, 2000]
    cols = plt.cm.viridis(np.linspace(0.08, 0.82, len(rates)))
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))
    fig.subplots_adjust(wspace=0.34, left=0.062, right=0.988, top=0.80, bottom=0.20)
    mins, errs, have = [], [], []
    for c, r in zip(cols, rates):
        f = os.path.join(RES, f"circles_N4_rate{r}_diag.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True); names = [str(x) for x in d["names"]]; t = d["t"]
        ds = float(d["ds"])
        hmin = None
        for a, b in itertools.combinations(names, 2):
            k = f"dist_{a}_{b}"
            if k in d.files:
                v = d[k] ** 2 - ds ** 2
                hmin = v if hmin is None else np.minimum(hmin, v)
        ax[0].plot(t, hmin, lw=1.7, color=c, label=f"{r} Hz")
        mins.append(np.nanmin(hmin)); have.append(r)
        errs.append(max(np.nanmax(d[f"{n}_path_err"][t > 10.5]) for n in names) * 100)
    ax[0].axhline(0, color=S.DANGER, lw=1.5)
    ax[0].set_xlabel("time [s]"); ax[0].set_ylabel(r"$\min_{ij}\,h_{ij}$  [m$^2$]")
    # the agents start far apart; on the full range the encounters are a flat line at the axis
    ax[0].set_ylim(-0.25, 3.0)
    ax[0]._qc_zoom = True          # the start transient is deliberately above the view
    S.panel_title(ax[0], "(a)  the barrier, one curve per rate")
    ax[0].legend(ncol=2, loc="upper right", fontsize=11); webqc.make_room(ax[0])

    # read as a distance: how far outside the separation sphere the closest approach was
    ds2 = 0.5 ** 2
    micro = [(np.sqrt(m + ds2) - 0.5) * 1e6 for m in mins]
    ax[1].semilogx(have, micro, "o-", lw=2.0, ms=8, color=S.CTRL["proposed"])
    ax[1].axhline(0, color=S.DANGER, lw=1.5)
    ax[1].set_ylim(0, max(micro) * 1.35)
    ax[1].set_xlabel("control rate [Hz]")
    ax[1].set_ylabel(r"closest approach past $d_s$  [$\mu$m]")
    S.panel_title(ax[1], "(b)  the tightest sampled margin")

    ax[2].semilogx(have, errs, "o-", lw=2.0, ms=8, color=S.CTRL["proposed"])
    ax[2].set_ylim(0, max(errs) * 1.35)
    ax[2].set_xlabel("control rate [Hz]"); ax[2].set_ylabel("settled path error [cm]")
    S.panel_title(ax[2], "(c)  path error after convergence")
    for a in ax[1:]:
        a.set_xticks(have); a.set_xticklabels([str(r) for r in have])
        a.minorticks_off()
    fig.suptitle("What the sampling period costs", x=0.062, ha="left", fontsize=15,
                 fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


# ---------------------------------------------------------------- solver cost

def fig_bench(out="W_bench", display_px=1180):
    b = json.load(open(os.path.join(RES, "bench", "bench.json")))
    runs = b["runs"]; Ns = [r["N"] for r in runs]
    KEYS = [("closed_form", S.CTRL["proposed"], "projection, closed form"),
            ("osqp_warm", S.CTRL["se3"], "OSQP, warm start, data updated"),
            ("osqp_solve", S.CTRL["baseline"], "OSQP, fresh factorization")]
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.4))
    fig.subplots_adjust(wspace=0.36, left=0.068, right=0.988, top=0.80, bottom=0.235)

    def stat(key, what):
        return np.array([r[key][what] * 1e6 for r in runs])

    for key, col, lab in KEYS:
        if key not in runs[0]:
            continue
        ax[0].fill_between(Ns, stat(key, "p05"), stat(key, "p95"), color=col, alpha=0.15, lw=0)
        ax[0].plot(Ns, stat(key, "median"), "o-", lw=2.0, ms=6.5, color=col, label=lab)
    ax[0].set_yscale("log")
    ax[0].set_xlabel("team size $N$"); ax[0].set_ylabel(r"time per agent per step  [$\mu$s]")
    S.panel_title(ax[0], "(a)  solving the safety program")

    for key, col, lab in KEYS:
        if key not in runs[0]:
            continue
        ax[1].plot(Ns, stat(key, "p95") / stat(key, "median"), "o-", lw=2.0, ms=6.5, color=col)
    ax[1].axhline(1.0, color=S.MUTED, lw=1.2, ls="--")
    ax[1].set_xlabel("team size $N$"); ax[1].set_ylabel("95th percentile / median")
    ax[1].set_ylim(0.9, None)
    S.panel_title(ax[1], "(b)  how predictable the cost is")

    data = [np.array(r["closed_form"]["samples"]) * 1e6 for r in runs]
    bp = ax[2].boxplot(data, positions=range(len(Ns)), widths=0.62, patch_artist=True,
                       showfliers=False, medianprops=dict(color="white", lw=1.6))
    for pt in bp["boxes"]:
        pt.set_facecolor(S.CTRL["proposed"]); pt.set_edgecolor("none")
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(S.MUTED)
    ax[2].set_xticks(range(len(Ns))); ax[2].set_xticklabels([str(n) for n in Ns])
    ax[2].set_xlabel("team size $N$"); ax[2].set_ylabel(r"projection time  [$\mu$s]")
    ax[2].set_ylim(0, None)
    S.panel_title(ax[2], f"(c)  the projection alone, {b['reps']} repetitions")

    fig.legend([Line2D([], [], color=c, lw=2.4) for _, c, _ in KEYS], [l for _, _, l in KEYS],
               ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11.5,
               frameon=False, handlelength=2.4, columnspacing=2.0)
    fig.suptitle("Cost of the safety step", x=0.068, ha="left", fontsize=15, fontweight="semibold",
                 color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


# ---------------------------------------------------------------- responsibility split, actuator box

def _series(tag, key, agent=None):
    d = np.load(os.path.join(RES, f"{tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]
    return d, names, d["t"]


def fig_weights(out="W_weights", display_px=1180):
    """A deliberately symmetric encounter, and what the responsibility split does to it."""
    cases = [("sine_N2_sym_w50", "equal split, $w_{ij}=1/2$", S.CTRL["proposed"]),
             ("sine_N2_sym_blend", "distance-blended split", S.CTRL["se3"]),
             ("sine_N2_sym_prio", "committed right of way", S.OK)]
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.6))
    fig.subplots_adjust(wspace=0.34, left=0.062, right=0.988, top=0.815, bottom=0.265)
    stopped = False
    for tag, lab, col in cases:
        f = os.path.join(RES, f"{tag}_diag.npz")
        if not os.path.exists(f):
            continue
        d, names, t = _series(tag, None)
        ds = float(d["ds"])
        a, b = names[0], names[1]
        ax[0].plot(t, d[f"dist_{a}_{b}"], lw=2.0, color=col, label=lab)
        for n, ls in zip(names, ("-", "--")):
            ax[1].plot(t, d[f"{n}_eta2"], lw=1.8, ls=ls, color=col)
        ax[2].semilogy(t, np.maximum(d[f"{names[0]}_width"], 1e-3), lw=1.8, color=col)
        # a run that ends early was stopped by the certificate; mark it so it does not read as a
        # curve that simply went missing
        js = os.path.join(RES, "summaries", tag.replace("sine_N2_", "") + ".json")
        if os.path.exists(js) and json.load(open(js))["infeasible"]:
            stopped = True
            for a_, y in ((ax[0], d[f"dist_{a}_{b}"][-1]), (ax[1], d[f"{names[0]}_eta2"][-1]),
                          (ax[2], max(d[f"{names[0]}_width"][-1], 1e-3))):
                a_.plot([t[-1]], [y], marker="X", ms=11, color=S.DANGER, mec="white", mew=1.2,
                        zorder=6, clip_on=False)
    ax[0].axhline(float(d["ds"]), color=S.DANGER, lw=1.5, label="separation $d_s$")
    ax[0].set_ylabel("pair distance [m]"); ax[0].set_ylim(0, None)
    S.panel_title(ax[0], "(a)  do they clear each other")
    ax[1].set_ylabel(r"along-path speed $\eta_2^i$  [m/s]")
    S.panel_title(ax[1], "(b)  both agents, solid and dashed")
    ax[2].set_ylabel(r"interval width  [m/s$^4$]")
    S.panel_title(ax[2], "(c)  authority left to the filter")
    for a_ in ax:
        a_.set_xlabel("time [s]")
        a_.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    h = [Line2D([], [], color=c, lw=2.2, label=l) for _, l, c in cases]
    h.append(Line2D([], [], color=S.DANGER, lw=1.5, label="separation $d_s$"))
    if stopped:
        h.append(Line2D([], [], color=S.DANGER, marker="X", ls="none", ms=9,
                        label="certificate stopped the run"))
    fig.legend(handles=h, ncol=4, loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11.5,
               frameon=False, handlelength=2.4, columnspacing=1.8)
    ax[0].get_legend().remove() if ax[0].get_legend() else None
    fig.suptitle("Two agents arriving at a crossing at the same speed", x=0.062, ha="left",
                 fontsize=15, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


def fig_actuator(out="W_actuator", display_px=1180):
    """Actuator bounds enter as more rows of the same interval form, so the test does not change."""
    cases = [("sine_N2_box_none", "no actuator bound", "#8A9199"),
             ("sine_N2_box_a", r"$|\tau_k|\leq 30$ mN m", S.CTRL["proposed"]),
             ("sine_N2_box_b", r"$|\tau_k|\leq 26$ mN m", S.CTRL["se3"]),
             ("sine_N2_box_c", r"$|\tau_k|\leq 23$ mN m", S.CTRL["baseline"]),
             ("sine_N2_box_d", r"$|\tau_k|\leq 21$ mN m", "#7A4FBF")]
    LEVELS = {"sine_N2_box_a": 30.0, "sine_N2_box_b": 26.0, "sine_N2_box_c": 23.0}
    S.use(1.0)
    fig, ax = plt.subplots(1, 3, figsize=(13.4, 4.3))
    fig.subplots_adjust(wspace=0.34, left=0.062, right=0.988, top=0.80, bottom=0.235)
    drawn = []
    for tag, lab, col in cases:
        f = os.path.join(RES, f"{tag}_diag.npz")
        if not os.path.exists(f):
            continue
        d = np.load(f, allow_pickle=True); names = [str(x) for x in d["names"]]; t = d["t"]
        n = names[0]
        # the bounds do their work in the first seconds, so each panel is cut to its own window
        w0 = t <= 12.0
        w1 = t <= 4.0
        ax[0].semilogy(t[w0], np.maximum(d[f"{n}_width"][w0], 1e-4), lw=1.8, color=col)
        ax[1].plot(t[w1], np.abs(d[f"{n}_nu"][w1, 1]) * 1e3, lw=1.8, color=col)
        if tag in LEVELS:
            ax[1].axhline(LEVELS[tag], color=col, lw=1.0, ls=":", alpha=0.85)
        ax[2].semilogy(t, np.maximum(d[f"{n}_path_err"] * 100, 1e-4), lw=1.8, color=col)
        js = os.path.join(RES, "summaries", tag.replace("sine_N2_", "").replace("circles_N4_", "")
                          + ".json")
        if os.path.exists(js) and json.load(open(js))["infeasible"]:
            for a_, y in ((ax[0], max(d[f"{n}_width"][w0][-1], 1e-4)),
                          (ax[1], abs(d[f"{n}_nu"][w1][-1, 1]) * 1e3),
                          (ax[2], max(d[f"{n}_path_err"][-1] * 100, 1e-4))):
                a_.plot([t[-1]], [y], marker="X", ms=11, color=S.DANGER, mec="white", mew=1.2,
                        zorder=6, clip_on=False)
        drawn.append((lab, col))
    ax[0].set_ylabel(r"interval width  [m/s$^4$]")
    S.panel_title(ax[0], "(a)  the bounds narrow the interval")
    ax[1].set_ylabel(r"roll torque $|\tau_1|$  [mN m]")
    S.panel_title(ax[1], "(b)  the applied torque respects them")
    ax[2].set_ylabel("distance to the path [cm]")
    S.panel_title(ax[2], "(c)  path following is unaffected")
    for a_ in ax:
        a_.set_xlabel("time [s]")
        a_.xaxis.set_major_locator(MaxNLocator(nbins=5, prune="lower"))
    fig.legend([Line2D([], [], color=c, lw=2.2) for _, c in drawn], [l for l, _ in drawn],
               ncol=len(drawn), loc="lower center", bbox_to_anchor=(0.5, -0.02), fontsize=11.5,
               frameon=False, handlelength=2.4, columnspacing=1.8)
    fig.suptitle("Actuator bounds as extra rows of the same scalar program", x=0.062, ha="left",
                 fontsize=15, fontweight="semibold", color=S.INK)
    webqc.save_gated(fig, OUT, out, display_px=display_px)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--which", default="all"); a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if a.which in ("all", "heat"):
        fig_heatmaps()
    if a.which in ("all", "mech"):
        fig_mechanism()
    if a.which in ("all", "rates"):
        fig_rates()
    if a.which in ("all", "bench"):
        fig_bench()
    if a.which in ("all", "weights"):
        fig_weights()
    if a.which in ("all", "actuator"):
        fig_actuator()
