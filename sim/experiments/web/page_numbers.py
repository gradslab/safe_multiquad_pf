r"""Collect every number the project page quotes into results/web/numbers.json.

The page reads this file, so a rerun that changes a result changes the page.
"""
import glob, itertools, json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..",
                                "src", "tflqp"))
import paths as _P

CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}
_PATHS = {}


def transverse_cm(base, sfx, tag, settle, scn):
    """The transverse output the manuscript reports: max over agents of |xi_1| x 100.

    Read from the state log through the path level sets rather than from the diagnostics, because
    the geometric baseline does not log transformed states.
    """
    f = os.path.join(RES, f"{base}{sfx}{tag}.npz")
    if not os.path.exists(f) or scn != "circles":
        return None
    S = np.load(f)
    t = S["t"]
    w = t > settle
    if not w.any():
        return None
    best = 0.0
    for n in [k for k in S.files if k.startswith("quad_")]:
        if n not in _PATHS:
            _PATHS[n] = _P.lifted_circle(n, *CEN[n], R=1.5)
        sv = np.array([_PATHS[n].s_val(y) for y in S[n][w][:, 6:9]])
        best = max(best, float(np.nanmax(np.abs(sv[:, 0]))) * 100.0)
    return best

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RES = os.path.join(ROOT, "results")
OUT = os.path.join(RES, "web")
os.makedirs(OUT, exist_ok=True)

# "after convergence" is a fixed instant per scenario, so all three controllers share the window
SET = {"circles": ("circles_N4", "_cs1p4", 12.0), "sine": ("sine_N2", "", 8.0)}
CTRL = {"proposed": "", "baseline": "_baseline", "se3": "_se3"}


def run_stats(base, sfx, tag, settle=10.5):
    f = os.path.join(RES, f"{base}{sfx}{tag}_diag.npz")
    if not os.path.exists(f):
        return None
    d = np.load(f, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    t = d["t"]; ds = float(d["ds"])
    w = t > min(settle, 0.5 * t[-1])
    out = {"t_end": float(t[-1]), "N": len(names), "ds": ds, "settle_s": float(min(settle, 0.5 * t[-1]))}

    def worst(key, scale=1.0, mask=w):
        vals = []
        for n in names:
            k = f"{n}_{key}"
            if k in d.files:
                v = np.asarray(d[k], float)
                v = v[mask] if v.ndim == 1 else v[mask, 0]
                v = v[np.isfinite(v)]
                if v.size:
                    vals.append(np.max(np.abs(v)) * scale)
        return float(max(vals)) if vals else None

    out["path_err_cm"] = worst("path_err", 100.0)
    yaw = []
    for n in names:
        k = f"{n}_mu"
        if k in d.files:
            v = np.degrees(np.asarray(d[k], float)[w, 0])
            v = v[np.isfinite(v)]
            if v.size:
                yaw.append(float(np.max(np.abs(v))))
    out["heading_deg"] = max(yaw) if yaw else None
    out["eq_resid"] = worst("eq_resid", mask=slice(None))     # "never exceeds", so whole run
    out["delta_star"] = worst("delta_star")
    smin = [float(np.nanmin(d[f"{n}_sigma_min_D"])) for n in names if f"{n}_sigma_min_D" in d.files]
    out["sigma_min_D"] = min(smin) if smin else None
    wid = [np.asarray(d[f"{n}_width"], float) for n in names if f"{n}_width" in d.files]
    if wid:
        allw = np.concatenate([v[np.isfinite(v)] for v in wid])
        out["interval_width_min"] = float(allw.min()) if allw.size else None
    thr = [float(np.nanmin(d[f"{n}_x13_margin"])) for n in names if f"{n}_x13_margin" in d.files]
    out["thrust_margin_N"] = min(thr) if thr else None
    hmin = None
    for a, b in itertools.combinations(names, 2):
        k = f"dist_{a}_{b}"
        if k in d.files:
            v = d[k] ** 2 - ds ** 2
            hmin = v if hmin is None else np.minimum(hmin, v)
    out["h_min"] = float(np.nanmin(hmin)) if hmin is not None else None
    out["dist_min"] = float(np.sqrt(out["h_min"] + ds ** 2)) if hmin is not None else None
    return out


def _encounters(diag_path, ds, thresh=2.0):
    """Pairwise encounters in a run: each interval where a pair closes to within thresh*d_s."""
    d = np.load(diag_path, allow_pickle=True)
    names = [str(x) for x in d["names"]]
    n = 0
    for a, b in itertools.combinations(names, 2):
        k = f"dist_{a}_{b}"
        if k not in d.files:
            continue
        close = d[k] < thresh * ds
        n += int(np.sum(np.diff(close.astype(int)) == 1)) + (1 if close[0] else 0)
    return n


def sweeps():
    out = {}
    tot_enc = tot_fail = 0
    for kind, key in (("sine", "ds"), ("circles", "center_scale")):
        rows = [json.load(open(f)) for f in glob.glob(os.path.join(RES, "sweeps", "symmetric",
                                                                   f"{kind}_a*.json"))]
        if not rows:
            continue
        stopped = [r for r in rows if r["infeasible"]]
        enc = 0
        base = "sine_N2" if kind == "sine" else "circles_N4"
        for r in rows:
            ax = r["ds"] if kind == "sine" else r["center_scale"]
            tag = f"_sw{kind}_{ax}_{r.get('v_rel', r['vscale'])}".replace(".", "p")
            dp = os.path.join(RES, f"{base}{tag}_diag.npz")
            if os.path.exists(dp):
                enc += _encounters(dp, r["ds"])
        tot_enc += enc; tot_fail += len(stopped)
        out[kind] = {
            "encounters_cleared": enc,
            "cells": len(rows),
            "ran": len(rows) - len(stopped),
            "stopped": len(stopped),
            "worst_h_over_all_cells": float(min(r["min_h"] for r in rows)),
            "median_stop_fraction": float(np.median([r["t_infeasible"] / r["tmax"]
                                                     for r in stopped])) if stopped else None,
            "axis": key,
            "axis_values": sorted({r[key] for r in rows}),
            "speed_values": sorted({r.get("v_rel", r["vscale"]) for r in rows}),
        }
    out["totals"] = {"encounters_cleared": tot_enc, "encounters_failed": tot_fail,
                     "encounters": tot_enc + tot_fail,
                     "rate": tot_enc / max(tot_enc + tot_fail, 1)}
    return out


def rates():
    out = []
    for r in (100, 200, 500, 1000, 2000):
        s = run_stats("circles_N4", "", f"_rate{r}", settle=10.5)
        if s:
            s["rate"] = r
            out.append(s)
    return out


def bench():
    f = os.path.join(RES, "bench", "bench.json")
    if not os.path.exists(f):
        return None
    b = json.load(open(f))
    rows = []
    for r in b["runs"]:
        rows.append({"N": r["N"], "rows": r["rows"],
                     "closed_us": r["closed_form"]["median"] * 1e6,
                     "osqp_us": r["osqp_solve"]["median"] * 1e6,
                     "osqp_setup_us": r["osqp_setup"]["median"] * 1e6,
                     "step_us": r["proposed_step"]["median"] * 1e6})
    ratios = [r["osqp_us"] / r["closed_us"] for r in rows]
    return {"hardware": b["hardware"], "reps": b["reps"], "rows": rows,
            "ratio_median": float(np.median(ratios)), "ratio_min": float(min(ratios)),
            "ratio_max": float(max(ratios))}


def main():
    doc = {"scenarios": {}, "sweeps": sweeps(), "rates": rates(), "bench": bench()}
    for scn, (base, tag, settle) in SET.items():
        doc["scenarios"][scn] = {}
        for c, sfx in CTRL.items():
            st = run_stats(base, sfx, tag, settle)
            if st is not None:
                st["transverse_cm"] = transverse_cm(base, sfx, tag, settle, scn)
            doc["scenarios"][scn][c] = st
    for name, tag in (("symmetric_equal", "sine_N2_sym_w50"), ("symmetric_blend", "sine_N2_sym_blend"),
                      ("symmetric_priority", "sine_N2_sym_prio")):
        s = run_stats(tag.rsplit("_sym", 1)[0], "", "_sym" + tag.split("_sym", 1)[1], 12.0)
        if s:
            doc.setdefault("symmetric", {})[name] = s
    for name, tag in (("none", "_box_none"), ("a", "_box_a"), ("b", "_box_b"),
                      ("c", "_box_c"), ("d", "_box_d")):
        s = run_stats("sine_N2", "", tag, 12.0)
        if s:
            doc.setdefault("actuator", {})[name] = s
    with open(os.path.join(OUT, "numbers.json"), "w") as fh:
        json.dump(doc, fh, indent=1)
    print(json.dumps({k: (list(v) if isinstance(v, dict) else v) for k, v in doc.items()
                      if k != "bench"}, indent=1)[:1200])
    print("wrote", os.path.join(OUT, "numbers.json"))


if __name__ == "__main__":
    main()
