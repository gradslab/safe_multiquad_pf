r"""Full simulation suite: every scenario, proposed and baseline, then figures and a Drake GIF.

  circles N=4   four intersecting lifted circles
  sine    N=2   mirrored nonplanar sinusoids (non-arc-length natural parameter)
  three   N=3   three lifted circles through a common point, 120 deg symmetry

Usage: python3 experiments/run_suite.py [--tmax 30] [--no-gif]
"""
import argparse, os, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SCEN = [("circles", 4), ("sine", 2), ("three", 3)]


def sh(cmd, tag):
    t0 = time.time()
    print(f"\n=== {tag} ===\n$ {' '.join(cmd)}", flush=True)
    r = subprocess.run(cmd, cwd=HERE, capture_output=True, text=True)
    out = (r.stdout or "") + (r.stderr or "")
    keep = [l for l in out.splitlines()
            if any(k in l for k in ("quad_", "min pairwise", "completed", "wrote", "QC ",
                                    "LAYOUT GATE", "Error", "Traceback", "INFEASIBLE"))]
    print("\n".join(keep[-16:]), flush=True)
    print(f"[{tag}] exit={r.returncode} in {time.time()-t0:.0f}s", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=30.0)
    ap.add_argument("--P", type=float, default=100.0)
    ap.add_argument("--no-gif", action="store_true")
    a = ap.parse_args()
    fails = []
    for sc, n in SCEN:
        for ctl in ("proposed", "baseline"):
            cmd = [sys.executable, "run_circles.py", "--scenario", sc, "--agents", str(n),
                   "--offpath", "--P", str(a.P), "--tmax", str(a.tmax), "--controller", ctl]
            if sh(cmd, f"{sc} N={n} {ctl}"): fails.append(f"{sc}/{ctl}")
        if sh([sys.executable, "make_figures.py", "--scenario", sc, "--N", str(n)],
              f"{sc} figures"): fails.append(f"{sc}/figures")
        if not a.no_gif:
            if sh([sys.executable, "drake_render.py", "--scenario", sc, "--agents", str(n),
                   "--tmax", str(a.tmax), "--fps", "20", "--capture-every", "25", "--P", str(a.P)],
                  f"{sc} GIF"): fails.append(f"{sc}/gif")
    print("\n" + "=" * 60)
    print("SUITE FAILURES:", ", ".join(fails) if fails else "none")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
