r"""Timing for the safety step.

Separates the per-agent control step, which both controllers pay alike, from the solve itself: the
closed-form projection against an OSQP solve of the same program. OSQP is timed through its own
interface, not through cvxpy, so it is charged for the solver and not for modelling overhead.
"""
import json, os, platform, subprocess, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "src", "tflqp"))
OUT = os.path.join(ROOT, "results", "bench")
os.makedirs(OUT, exist_ok=True)

import numpy as np
import paths as P
import dynamics as dyn
import controller as ctl
import optimizer as opt
import tfl
import scenario_circles as scn
import osqp
from scipy import sparse

NS = [2, 3, 4, 6, 8, 10, 12, 14, 16]
REPS = 400


def hardware():
    info = {"python": sys.version.split()[0], "platform": platform.platform(),
            "processor": platform.processor() or platform.machine(),
            "numpy": np.__version__, "osqp": osqp.__version__}
    try:
        info["cpu"] = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"],
                                              text=True).strip()
        info["cores"] = subprocess.check_output(["sysctl", "-n", "hw.ncpu"], text=True).strip()
    except Exception:
        pass
    return info


def build_team(N, model):
    """N agents on distinct intersecting circles, laid out on a grid so neighbours are in range."""
    side = int(np.ceil(np.sqrt(N)))
    gains = scn.default_gains()
    cfgs, X14 = [], []
    rng = np.random.default_rng(7)
    for k in range(N):
        cx, cy = 1.0 * (k % side), 1.0 * (k // side)
        pth = P.lifted_circle(f"q{k}", cx, cy, R=1.5)
        v = 0.45 if k % 2 == 0 else -0.45
        c = ctl.AgentConfig(f"q{k}", pth, v_des=v, psi_des=0.0, gains=gains, idx=k,
                            fmin=0.05 * scn.M_HOVER, ds=0.5, lam_att=10.0, lam_pair=3.75,
                            lam_f=(10.0, 10.0), P=100.0)
        q0 = 2 * np.pi * k / max(N, 1)
        c.q0 = q0
        pos = pth.sig(q0, 0); tan = pth.sig(q0, 1); tan = tan / np.linalg.norm(tan)
        x = np.zeros(14)
        x[6:9] = pos + 0.01 * rng.standard_normal(3)
        x[9:12] = v * tan
        x[12] = scn.M_HOVER
        cfgs.append(c); X14.append(x)
    return cfgs, X14


def bench_N(N, model, reps=REPS):
    cfgs, X14 = build_team(N, model)
    kin = [ctl.kinematics(x, model) for x in X14]
    qw = [c.q0 for c in cfgs]

    # warm up the lambdified path callables and the JIT-free numpy paths
    for i in range(N):
        ctl.step(i, X14, kin, cfgs, model, qw)

    # ---- per-agent controller time, proposed
    t_prop = []
    for _ in range(reps):
        t0 = time.perf_counter()
        ctl.step(0, X14, kin, cfgs, model, qw)
        t_prop.append(time.perf_counter() - t0)

    # ---- assemble one representative program to time the solves in isolation
    cfg = cfgs[0]; x = X14[0]
    R = tfl.assemble(x, cfg.path, model, cfg.gains, cfg.v_des, cfg.psi_des, qw[0])
    import barriers as bar
    rows = bar.attitude_rows(x, model, cfg.eps, cfg.lam_att, cfg.lam_att)
    rows.append(bar.thrust_row(x, cfg.fmin, *cfg.lam_f))
    rows += bar.speed_rows(R.eta, R.L4_beta1, R.D[2, :], cfg.v_max, cfg.lam_v)
    Vi, Bqi = kin[0]
    for j in range(1, N):
        Vj, _ = kin[j]
        rows.append(bar.collision_row(x, X14[j], Vi, Vj, Bqi, cfg.ds, [cfg.lam_pair] * 4, 0.5))
    A = np.array([r["a"] for r in rows]); b = np.array([r["b"] for r in rows])

    t_closed = []
    for _ in range(reps):
        t0 = time.perf_counter()
        opt.reduce_and_solve(A, b, R.nu_tfl, R.p, cfg.W, cfg.P)
        t_closed.append(time.perf_counter() - t0)

    # ---- the same program solved numerically: min 1/2||nu-nu_tfl||_W^2  s.t.  A nu >= -b
    Pm = sparse.csc_matrix(cfg.W)
    q = -cfg.W @ R.nu_tfl
    Am = sparse.csc_matrix(A)
    lo = -b; hi = np.full(len(b), np.inf)
    t_osqp, t_osqp_setup, t_osqp_warm = [], [], []
    for _ in range(reps):
        t0 = time.perf_counter()
        m = osqp.OSQP()
        m.setup(P=Pm, q=q, A=Am, l=lo, u=hi, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
        t1 = time.perf_counter()
        m.solve()
        t_osqp.append(time.perf_counter() - t1)
        t_osqp_setup.append(t1 - t0)
    # the fastest legitimate way to run OSQP in a loop: keep the workspace, update the data
    mw = osqp.OSQP()
    mw.setup(P=Pm, q=q, A=Am, l=lo, u=hi, verbose=False, eps_abs=1e-8, eps_rel=1e-8)
    mw.solve()
    for _ in range(reps):
        t0 = time.perf_counter()
        mw.update(Ax=Am.data, l=lo, q=q)
        mw.solve()
        t_osqp_warm.append(time.perf_counter() - t0)

    st = lambda v: {"mean": float(np.mean(v)), "std": float(np.std(v)), "median": float(np.median(v)),
                    "p05": float(np.percentile(v, 5)), "p95": float(np.percentile(v, 95)),
                    "min": float(np.min(v)), "n": len(v), "samples": [float(z) for z in v]}
    return {"N": N, "rows": len(rows), "proposed_step": st(t_prop), "closed_form": st(t_closed),
            "osqp_solve": st(t_osqp), "osqp_setup": st(t_osqp_setup), "osqp_warm": st(t_osqp_warm)}


def main():
    model = dyn.Model()
    res = {"hardware": hardware(), "reps": REPS, "runs": []}
    for N in NS:
        r = bench_N(N, model)
        res["runs"].append(r)
        print(f"N={N:2d} rows={r['rows']:3d} | controller step {r['proposed_step']['mean']*1e6:8.1f} us"
              f" | closed form {r['closed_form']['mean']*1e6:7.2f} us"
              f" | OSQP solve {r['osqp_solve']['mean']*1e6:8.1f} us"
              f" (setup {r['osqp_setup']['mean']*1e6:7.1f}, warm {r['osqp_warm']['median']*1e6:7.1f})"
              f" | ratio {r['osqp_solve']['median']/r['closed_form']['median']:6.1f}x")
    with open(os.path.join(OUT, "bench.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    print("wrote", os.path.join(OUT, "bench.json"))


if __name__ == "__main__":
    main()
