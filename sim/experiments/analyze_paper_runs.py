r"""Uniform, state-log-based analysis of the production runs.

All metrics are computed from the raw state logs (positions, yaw) identically for every controller:
  - distance to the assigned path (Newton nearest-point projection, warm-started along the run)
  - yaw deviation |psi - psi_des|
  - yaw smoothness through encounter windows (zero crossings of the yaw rate, max |yaw rate|)
  - min pairwise distance / barrier h
  - post-convergence window: t in [T_SETTLE, Tmax]  (stated in Sec. VI)
Plus, for the proposed controller only (from its diagnostic log): equality residual, strict feasibility,
min thrust (Assumption 1), delta* stats.

Usage: python3 experiments/analyze_paper_runs.py [--tag _paper] [--json out.json]
"""
import argparse
import itertools
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "src", "tflqp")))
RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))

import scenario_circles as circ
import scenario_sine as sine

T_SETTLE = 15.0          # stated post-convergence window start (both scenarios)
ENC_DIST = 0.6           # encounter window: some pair within this distance [m]
DS = 0.5


def paths_for(scenario, names):
    if scenario == "circles":
        return {n: circ.make_configs(names=tuple(names))[i].path for i, n in enumerate(names)}
    return {n: sine.make_path(n) for n in names}


def q0_for(scenario, names):
    if scenario == "circles":
        return {n: circ.Q0[n] for n in names}
    return {n: sine.X0_START[n] for n in names}


def project_series(path, Y, q0):
    """Distance to path for a (T,3) position series, warm-started Newton projection."""
    q = q0
    out = np.empty(len(Y))
    for k, y in enumerate(Y):
        q = path.q_star(y, q)
        out[k] = np.linalg.norm(y - path.sig(q, 0))
    return out


def yaw_rate_zero_crossings(psi, dt, window):
    """Count sign changes of the (lightly smoothed) yaw rate within a boolean window."""
    rate = np.gradient(psi, dt)
    k = max(1, int(0.05 / dt))          # 50 ms boxcar
    kern = np.ones(k) / k
    rate = np.convolve(rate, kern, mode="same")
    r = rate[window]
    r = r[np.abs(r) > 1e-4]             # ignore numerical dither around zero
    return int(np.sum(np.abs(np.diff(np.sign(r))) > 0)), float(np.nanmax(np.abs(rate[window])) if window.any() else np.nan)


def analyze(scenario, N, ctag, tag):
    f = os.path.join(RESULTS, f"{scenario}_N{N}{ctag}{tag}.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    t = d["t"]; dt = float(np.median(np.diff(t)))
    names = [k for k in d.files if k.startswith("quad")]
    P = paths_for(scenario, names)
    Q0 = q0_for(scenario, names)
    w = t >= T_SETTLE

    res = {"file": os.path.basename(f), "t_end": float(t[-1]), "n_steps": int(len(t))}
    # pairwise distances + encounter windows
    pos = {n: d[n][:, 6:9] for n in names}
    mind = np.full(len(t), np.inf)
    for a, b in itertools.combinations(names, 2):
        dist = np.linalg.norm(pos[a] - pos[b], axis=1)
        mind = np.minimum(mind, dist)
    enc = mind < ENC_DIST
    res["min_pair_dist"] = float(mind.min()) if np.isfinite(mind).any() else None
    res["min_h"] = float((mind.min() ** 2 - DS ** 2)) if np.isfinite(mind).any() else None
    res["encounter_frac_post"] = float(np.mean(enc[w]))

    per = {}
    for n in names:
        pe = project_series(P[n], pos[n], Q0[n])
        psi = d[n][:, 2]
        zc, maxrate = yaw_rate_zero_crossings(psi, dt, enc & w) if enc.any() else (0, np.nan)
        per[n] = {
            "max_path_dev_post": float(np.nanmax(pe[w])),
            "max_yaw_dev_post_deg": float(np.degrees(np.nanmax(np.abs(psi[w])))),
            "yawrate_zero_crossings_in_encounters": zc,
            "max_yawrate_in_encounters_degps": float(np.degrees(maxrate)) if np.isfinite(maxrate) else None,
        }
    res["agents"] = per
    res["max_path_dev_post"] = max(a["max_path_dev_post"] for a in per.values())
    res["max_yaw_dev_post_deg"] = max(a["max_yaw_dev_post_deg"] for a in per.values())

    # proposed-only theory diagnostics from the diag npz
    fd = os.path.join(RESULTS, f"{scenario}_N{N}{ctag}{tag}_diag.npz")
    if ctag == "" and os.path.exists(fd):
        g = np.load(fd, allow_pickle=True)
        eq = np.nanmax([np.nanmax(g[f"{n}_eq_resid"]) for n in names])
        wid = np.nanmin([np.nanmin(g[f"{n}_width"]) for n in names])
        x13 = np.nanmin([np.nanmin(g[f"{n}_x13"]) for n in names])
        feas = all(bool(np.all(g[f"{n}_feasible"])) for n in names)
        dstar = np.nanmax([np.nanmax(np.abs(np.nan_to_num(g[f"{n}_delta_star"]))) for n in names])
        res["proposed_checks"] = {"max_eq_resid": float(eq), "min_interval_width": float(wid),
                                  "min_thrust": float(x13), "always_feasible": feas,
                                  "max_abs_delta_star": float(dstar)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="_paper")
    ap.add_argument("--json", default="")
    args = ap.parse_args()
    out = {}
    for scenario, N in (("circles", 4), ("sine", 2)):
        for ctag, label in (("", "proposed"), ("_baseline", "tfl_cascade"), ("_se3", "se3_cascade")):
            r = analyze(scenario, N, ctag, args.tag)
            if r is None:
                print(f"[missing] {scenario} {label}")
                continue
            out[f"{scenario}_{label}"] = r
            print(f"== {scenario} {label} ({r['file']}), post-convergence window [{T_SETTLE}s, {r['t_end']:.0f}s]")
            print(f"   min pair dist {r['min_pair_dist']:.4f} m (h_min {r['min_h']:+.4f}) | "
                  f"max path dev {1e3*r['max_path_dev_post']:.2f} mm | max yaw dev {r['max_yaw_dev_post_deg']:.3f} deg")
            for n, a in r["agents"].items():
                print(f"   {n}: path {1e3*a['max_path_dev_post']:.2f} mm | yaw {a['max_yaw_dev_post_deg']:.3f} deg | "
                      f"yaw-rate zero-crossings in encounters {a['yawrate_zero_crossings_in_encounters']} "
                      f"(max rate {a['max_yawrate_in_encounters_degps']} deg/s)")
            if "proposed_checks" in r:
                c = r["proposed_checks"]
                print(f"   [theory] eq_resid {c['max_eq_resid']:.1e} | min width {c['min_interval_width']:.2e} | "
                      f"min thrust {c['min_thrust']:.2f} N | always feasible {c['always_feasible']} | "
                      f"max|delta*| {c['max_abs_delta_star']:.2e}")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(out, fh, indent=1)
        print("wrote", args.json)


if __name__ == "__main__":
    main()
