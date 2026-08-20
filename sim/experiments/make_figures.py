r"""Figure suite for the MINNORM v2 controller. House style follows the V1/V2 figures:
Paul Tol colourblind-safe palette keyed by agent, 9 pt type, lettered panels, desired paths dashed,
proposed solid and baseline dashed.

  F1 traj3d      3D trajectories over the assigned paths, off-path starts marked
  F2 channels    (a) path error (b) eta2 vs v_des (c) heading error (d) pairwise distance + d_s
                 (e) delta* (f) feasible interval (g) interval width (h) sigma_min(D), sigma_min(J_gamma)
  F3 collision   the collision barrier only: h_ij and its recursive coordinates Psi_{ij,0..3}
  F4 inputs      applied nu against the nominal TFL input nu_0
  F5 transverse  the hard chains xi_1..4, zeta_1..4
  F6 authorities row authorities abar_k; zero crossings are where the speed channel loses grip
  F7 baseline    proposed vs full-input filter: path error, min distance, top view

Usage: python3 experiments/make_figures.py --scenario circles --N 4
"""
import argparse, itertools, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D  # noqa
import qc_figures

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp"))
sys.path.insert(0, PKG)
import paths as P
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
FIG = os.path.join(RES, "figures"); os.makedirs(FIG, exist_ok=True)

COL = {"quad_A": "#4477AA", "quad_B": "#EE6677", "quad_C": "#228833", "quad_D": "#CCBB44"}
CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "legend.frameon": False,
                     "axes.grid": True, "grid.alpha": 0.25, "grid.linewidth": 0.5})


QC_ISSUES = []


def save(fig, name):
    """Save, but only after the layout gate has inspected the live figure."""
    issues = qc_figures.check(fig, name)
    QC_ISSUES.extend(issues)
    fig.savefig(os.path.join(FIG, name + ".pdf"), bbox_inches="tight")
    fig.savefig(os.path.join(FIG, name + ".png"), dpi=200, bbox_inches="tight")
    plt.close(fig); print("   ", name)


def get_path(scenario, n):
    if scenario == "sine":
        import scenario_sine as ssc; return ssc.make_path(n)
    if scenario == "three":
        import scenario_three as s3; return s3.make_path(n)
    return P.lifted_circle(n, *CEN[n], R=1.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", choices=("circles", "sine", "three"), default="circles")
    ap.add_argument("--N", type=int, default=4)
    a = ap.parse_args(); sc, N = a.scenario, a.N
    tag = f"{sc}_N{N}"
    st = np.load(os.path.join(RES, f"{tag}.npz"))
    dg = np.load(os.path.join(RES, f"{tag}_diag.npz"), allow_pickle=True)
    t = dg["t"]; names = [str(n) for n in dg["names"]]; ds = float(dg["ds"])
    vdes = {n: float(dg["v_des"][i]) for i, n in enumerate(names)}
    hasb = os.path.exists(os.path.join(RES, f"{tag}_baseline_diag.npz"))
    if hasb:
        db = np.load(os.path.join(RES, f"{tag}_baseline_diag.npz"), allow_pickle=True)
        sb = np.load(os.path.join(RES, f"{tag}_baseline.npz"))

    # ---------------- F1 3D trajectories ----------------
    fig = plt.figure(figsize=(5.2, 4.4)); ax = fig.add_subplot(111, projection="3d")
    ax.set_box_aspect((1, 1, 0.5), zoom=1.15)
    if sc == "sine":
        xs = np.concatenate([st[n][:, 6] for n in names]); qq = np.linspace(xs.min()-1, xs.max()+1, 300)
    else:
        qq = np.linspace(0, 2*np.pi, 320)
    for n in names:
        d = np.array([get_path(sc, n).sig(q, 0) for q in qq])
        ax.plot(d[:, 0], d[:, 1], d[:, 2], color=COL[n], ls="--", lw=0.7, alpha=0.55)
        S = st[n]; ax.plot(S[:, 6], S[:, 7], S[:, 8], color=COL[n], lw=1.3, label=n)
        ax.scatter(S[0, 6], S[0, 7], S[0, 8], color=COL[n], marker="o", s=22, depthshade=False)
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]"); ax.set_zlabel("z [m]")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.set_title("trajectories (solid) over assigned paths (dashed); ● off-path start")
    save(fig, f"F1_traj3d_{tag}")

    # ---------------- F2 channels ----------------
    fig, ax = plt.subplots(2, 4, figsize=(14, 5.6)); fig.subplots_adjust(wspace=0.30, hspace=0.34)
    for n in names: ax[0, 0].semilogy(t, dg[f"{n}_path_err"]*100, color=COL[n], lw=1.0, label=n)
    ax[0, 0].set_title(r"(a) path error dist$(x_q,\gamma)$"); ax[0, 0].set_ylabel("[cm]")
    ax[0, 0].legend(fontsize=7, ncol=2)
    for n in names:
        ax[0, 1].plot(t, dg[f"{n}_eta2"], color=COL[n], lw=1.0)
        ax[0, 1].axhline(vdes[n], color=COL[n], ls=":", lw=0.8, alpha=0.8)
    ax[0, 1].axhline(0, color="0.6", lw=0.6); ax[0, 1].set_title(r"(b) $\eta_2$ vs $v_{des}$ (dotted)")
    ax[0, 1].set_ylabel("[m/s]")
    for n in names: ax[0, 2].plot(t, dg[f"{n}_mu"][:, 0], color=COL[n], lw=1.0)
    ax[0, 2].axhline(0, color="0.6", lw=0.6); ax[0, 2].set_title(r"(c) heading error $e_\mu$")
    ax[0, 2].set_ylabel("[rad]")
    for p, q in itertools.combinations(names, 2):
        ax[0, 3].plot(t, dg[f"dist_{p}_{q}"], lw=0.85, label=f"{p[-1]}–{q[-1]}")
    ax[0, 3].axhline(ds, color="crimson", lw=1.0, label=r"$d_s$")
    ax[0, 3].set_title("(d) pairwise distance"); ax[0, 3].set_ylabel("[m]"); ax[0, 3].legend(fontsize=6, ncol=2)
    for n in names: ax[1, 0].plot(t, dg[f"{n}_delta_star"], color=COL[n], lw=1.0)
    ax[1, 0].axhline(0, color="0.6", lw=0.6); ax[1, 0].set_title(r"(e) $\delta^\star$ [m/s$^4$]")
    n0 = names[1]
    lo = np.clip(dg[f"{n0}_delta_lo"], -50, 50); hi = np.clip(dg[f"{n0}_delta_hi"], -50, 50)
    ax[1, 1].fill_between(t, lo, hi, color=COL[n0], alpha=0.22, lw=0)
    ax[1, 1].plot(t, dg[f"{n0}_delta_star"], color=COL[n0], lw=1.0, label=r"$\delta^\star$")
    if f"{n0}_delta_hat" in dg.files:
        ax[1, 1].plot(t, dg[f"{n0}_delta_hat"], color="0.25", lw=0.8, ls=":", label=r"$\hat\delta$")
    ax[1, 1].set_title(rf"(f) feasible interval, {n0} (clip $\pm50$)"); ax[1, 1].legend(fontsize=6)
    for n in names: ax[1, 2].plot(t, np.clip(dg[f"{n}_width"], 0, 60), color=COL[n], lw=1.0)
    ax[1, 2].set_title(r"(g) interval width $\delta_{hi}-\delta_{lo}$ (clip 60)")
    for n in names:
        ax[1, 3].plot(t, dg[f"{n}_sigma_min_D"], color=COL[n], lw=1.0)
        ax[1, 3].plot(t, dg[f"{n}_sigma_min_Jgamma"], color=COL[n], lw=0.8, ls="--")
    ax[1, 3].set_title(r"(h) $\sigma_{\min}(D)$ solid, $\sigma_{\min}(J_\gamma)$ dashed")
    for x_ in ax.flat: x_.set_xlabel("t [s]"); x_.tick_params(labelsize=8)
    save(fig, f"F2_channels_{tag}")

    # ---------------- F3 collision barrier only ----------------
    fig, ax = plt.subplots(1, 3, figsize=(12, 3.2)); fig.subplots_adjust(wspace=0.26)
    for p, q in itertools.combinations(names, 2):
        ax[0].plot(t, dg[f"dist_{p}_{q}"]**2 - ds**2, lw=0.95, label=f"{p[-1]}–{q[-1]}")
    ax[0].axhline(0, color="crimson", lw=1.0)
    ax[0].set_title(r"(a) $h_{ij}=\|x_{ij}\|^2-d_s^2$"); ax[0].set_ylabel(r"[m$^2$]")
    ax[0].legend(fontsize=6, ncol=3)
    mind = np.min([dg[k] for k in dg.files if k.startswith("dist_")], axis=0)
    ax[1].plot(t, mind, color="#4477AA", lw=1.1)
    ax[1].axhline(ds, color="crimson", lw=1.0, label=r"$d_s$")
    ax[1].set_title("(b) minimum pairwise distance"); ax[1].set_ylabel("[m]"); ax[1].legend(fontsize=7)
    for n in names:
        cp = dg[f"{n}_coll_Psi_min"]
        for k in range(cp.shape[1]):
            ax[2].plot(t, cp[:, k], color=COL[n], lw=0.8, alpha=1.0 - 0.18*k)
    ax[2].axhline(0, color="crimson", lw=1.0)
    ax[2].set_title(r"(c) recursive coordinates $\Psi_{ij0..ij3}$ (min over neighbours)")
    for x_ in ax: x_.set_xlabel("t [s]")
    save(fig, f"F3_collision_{tag}")

    # ---------------- F4 inputs ----------------
    fig, ax = plt.subplots(1, 4, figsize=(14, 2.9)); fig.subplots_adjust(wspace=0.30)
    lab = [r"$u_d$ [N/s$^2$]", r"$\tau_1$ [N m]", r"$\tau_2$ [N m]", r"$\tau_3$ [N m]"]
    for k in range(4):
        for n in names:
            ax[k].plot(t, dg[f"{n}_nu"][:, k], color=COL[n], lw=0.9)
            ax[k].plot(t, dg[f"{n}_nu_tfl"][:, k], color=COL[n], lw=0.7, ls=":", alpha=0.7)
        ax[k].set_title(f"({chr(97+k)}) " + lab[k]); ax[k].set_xlabel("t [s]")
    fig.suptitle(r"applied $\nu^\star$ (solid) vs nominal TFL $\nu_0$ (dotted)", y=1.04)
    save(fig, f"F4_inputs_{tag}")

    # ---------------- F5 transverse chains ----------------
    fig, ax = plt.subplots(2, 4, figsize=(14, 4.6)); fig.subplots_adjust(wspace=0.28, hspace=0.36)
    for j in range(4):
        for n in names:
            ax[0, j].plot(t, dg[f"{n}_xi"][:, j], color=COL[n], lw=0.9)
            ax[1, j].plot(t, dg[f"{n}_zeta"][:, j], color=COL[n], lw=0.9)
        ax[0, j].set_title(rf"$\xi_{j+1}$"); ax[1, j].set_title(rf"$\zeta_{j+1}$")
        ax[1, j].set_xlabel("t [s]")
        for r_ in (0, 1): ax[r_, j].axhline(0, color="0.6", lw=0.6)
    fig.suptitle(r"hard transverse chains $\chi=(\xi,\zeta)\to 0$", y=1.00)
    save(fig, f"F5_transverse_{tag}")

    # ---------------- F6 row authorities ----------------
    if f"{names[0]}_abar_coll" in dg.files:
        fig, ax = plt.subplots(1, len(names), figsize=(3.4*len(names), 2.9), sharey=True)
        fig.subplots_adjust(wspace=0.16)
        for i, n in enumerate(names):
            A = ax[i]
            ac = dg[f"{n}_abar_coll"]
            for c in range(ac.shape[1]):
                A.plot(t, ac[:, c], color=COL[n], lw=0.8, alpha=1.0-0.22*c)
            A.plot(t, dg[f"{n}_abar_thrust"], color="0.25", lw=0.9, ls="-.")
            A.plot(t, dg[f"{n}_abar_speed"][:, 0], color="crimson", lw=1.0, ls="--")
            A.axhline(0, color="0.4", lw=0.7)
            A.set_title(n); A.set_xlabel("t [s]")
        ax[0].set_ylabel(r"$\bar a_k$")
        ax[0].legend(handles=[Line2D([], [], color="0.5", lw=0.8, label="collision"),
                              Line2D([], [], color="0.25", lw=0.9, ls="-.", label="thrust"),
                              Line2D([], [], color="crimson", lw=1.0, ls="--", label=r"speed ($\equiv+1$)")],
                     fontsize=6, loc="upper right")
        fig.suptitle("row authorities: collision crosses zero at every encounter, speed never does", y=1.05)
        save(fig, f"F6_authorities_{tag}")

    # ---------------- F7 proposed vs baseline ----------------
    if hasb:
        fig, ax = plt.subplots(1, 3, figsize=(11.5, 3.2)); fig.subplots_adjust(wspace=0.28)
        for n in names:
            ax[0].semilogy(t, dg[f"{n}_path_err"]*100, color=COL[n], lw=1.1)
            ax[0].semilogy(db["t"], db[f"{n}_path_err"]*100, color=COL[n], lw=0.9, ls="--", alpha=0.85)
        ax[0].set_title(r"(a) path error dist$(x_q,\gamma)$"); ax[0].set_ylabel("[cm]")
        ax[0].legend(handles=[Line2D([], [], color="0.3", lw=1.1, label="proposed"),
                              Line2D([], [], color="0.3", lw=0.9, ls="--", label="full-input filter")],
                     fontsize=7, loc="lower left")
        mb = np.min([db[k] for k in db.files if k.startswith("dist_")], axis=0)
        ax[1].plot(t, mind, color="#4477AA", lw=1.1, label="proposed")
        ax[1].plot(db["t"], mb, color="#EE6677", lw=1.0, ls="--", label="full-input filter")
        ax[1].axhline(ds, color="crimson", lw=1.0, ls=":", label=r"$d_s$")
        ax[1].set_title("(b) minimum pairwise distance"); ax[1].set_ylabel("[m]"); ax[1].legend(fontsize=7)
        for n in names:
            d = np.array([get_path(sc, n).sig(q, 0) for q in qq])
            ax[2].plot(d[:, 0], d[:, 1], color=COL[n], ls=":", lw=0.8, alpha=0.65)
            ax[2].plot(st[n][:, 6], st[n][:, 7], color=COL[n], lw=1.1)
            ax[2].plot(sb[n][:, 6], sb[n][:, 7], color=COL[n], lw=0.9, ls="--", alpha=0.85)
        ax[2].set_aspect("equal", adjustable="box")
        ax[2].set_title("(c) top view: baseline leaves its path"); ax[2].set_xlabel("x [m]"); ax[2].set_ylabel("y [m]")
        for x_ in ax[:2]: x_.set_xlabel("t [s]")
        save(fig, f"F7_baseline_{tag}")


if __name__ == "__main__":
    main()
    if QC_ISSUES:
        print(f"\nLAYOUT GATE: {len(QC_ISSUES)} issue(s) across the suite")
        sys.exit(1)
    print("\nLAYOUT GATE: all figures clean")
