r"""Search desired speeds for the configuration where the proposed controller separates most clearly
from the two baselines.

For each candidate we run all three controllers from identical initial conditions and score:
  path error ratio   = max_post |dist(x_q,gamma)| baseline / proposed      (want large)
  yaw deviation      = max_post |psi| for each baseline                     (want large)
subject to the proposed run staying feasible for the whole horizon and every run being collision-free.
Nothing is reported unless the proposed run completes; a configuration that only looks good because the
proposed controller quit early is not a result.
"""
import argparse, itertools, os, subprocess, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp")); sys.path.insert(0, PKG)
RES = os.path.abspath(os.path.join(HERE, "..", "results"))
import paths as P
CEN = {"quad_A": (0, 0), "quad_B": (0, 1), "quad_C": (1, 0), "quad_D": (1, 1)}


def get_path(sc, n):
    if sc == "sine":
        import scenario_sine as s; return s.make_path(n)
    return P.lifted_circle(n, *CEN[n], R=1.5)


def run(sc, N, ctl, vdes, tmax, tag):
    cmd = [sys.executable, "run_circles.py", "--scenario", sc, "--agents", str(N), "--offpath",
           "--tmax", str(tmax), "--P", "100", "--lam-v", "20", "--controller", ctl, "--tag", tag]
    if vdes:
        cmd += ["--vdes"] + [str(v) for v in vdes]
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    return "INFEASIBLE at" not in (r.stdout + r.stderr)


def metrics(sc, N, tag, t_settle):
    f = os.path.join(RES, f"{sc}_N{N}{tag}.npz")
    if not os.path.exists(f):
        return None
    S = np.load(f); d = np.load(os.path.join(RES, f"{sc}_N{N}{tag}_diag.npz"), allow_pickle=True)
    names = [str(x) for x in d["names"]]; t = S["t"]; w = t >= t_settle
    if w.sum() < 10:
        return None
    perr = 0.0; yaw = 0.0
    for n in names:
        pth = get_path(sc, n)
        sv = np.array([pth.s_val(y) for y in S[n][w, 6:9]])
        perr = max(perr, np.abs(sv).max())
        yaw = max(yaw, np.abs(np.degrees(S[n][w, 2])).max())
    mb = None
    for a, b in itertools.combinations(names, 2):
        dd = S[a][:, 6:9] - S[b][:, 6:9]
        v = np.einsum("ij,ij->i", dd, dd) - 0.25
        mb = v if mb is None else np.minimum(mb, v)
    return {"path_err": perr, "yaw": yaw, "min_h": float(mb.min()), "T": float(t[-1])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", default="circles"); ap.add_argument("--N", type=int, default=4)
    ap.add_argument("--tmax", type=float, default=30.0)
    ap.add_argument("--settle", type=float, default=12.0)
    ap.add_argument("--scales", type=float, nargs="*", default=[1.0, 1.3, 1.6, 2.0])
    a = ap.parse_args()
    # read the nominal speeds from the scenario itself; hardcoding them here once produced a sine run
    # with quad_B travelling backwards, so the two agents never met and every ratio came out 1x.
    if a.scenario == "sine":
        import scenario_sine as _s
        cfgs = _s.make_configs()
    else:
        import scenario_circles as _s
        cfgs = _s.make_configs(names=list(_s.CENTERS)[:a.N]) if hasattr(_s, "CENTERS") else _s.make_configs()
    base = [c.v_des for c in cfgs][:a.N]
    print(f"nominal speeds from scenario_{a.scenario}: {base}")
    print(f"{'speeds':32s} {'proposed':>10s} {'TFL+filter':>22s} {'SE(3)+filter':>22s}")
    print(f"{'':32s} {'err[cm]':>10s} {'err[cm]':>10s} {'ratio':>5s} {'yaw':>5s} "
          f"{'err[cm]':>10s} {'ratio':>5s} {'yaw':>5s}")
    print("-" * 92)
    for sc_ in a.scales:
        v = [round(x * sc_, 3) for x in base]
        tag = f"_cs{str(sc_).replace('.','p')}"
        ok = run(a.scenario, a.N, "proposed", v, a.tmax, tag)
        mp = metrics(a.scenario, a.N, tag, a.settle)
        if not ok or mp is None:
            print(f"{str(v):32s}  proposed INFEASIBLE / no data -> skipped")
            continue
        row = f"{str(v):32s} {mp['path_err']*100:10.4f}"
        for ctl, sfx in (("baseline", "_baseline"), ("se3", "_se3")):
            run(a.scenario, a.N, ctl, v, a.tmax, tag)
            mb = metrics(a.scenario, a.N, sfx + tag, a.settle)
            if mb is None:
                row += f" {'n/a':>10s} {'':>5s} {'':>5s}"; continue
            row += (f" {mb['path_err']*100:10.3f} {mb['path_err']/max(mp['path_err'],1e-12):5.0f}x"
                    f" {mb['yaw']:5.1f}")
        print(row + f"   [proposed min_h={mp['min_h']:+.4f}]")


if __name__ == "__main__":
    main()
