r"""Feasibility sweeps over speed against geometry.

Each cell is a Drake rollout that either runs to t_max or stops the step the closed-form certificate
reports an empty interval. There is no fallback controller, so that stop is the only failure mode.
The horizon is scaled with speed, so every cell covers the same path distance.

    python3 experiments/web/sweep.py --grid sine
    python3 experiments/web/sweep.py --grid circles
"""
import argparse, itertools, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(ROOT, "experiments", "run_circles.py")
OUT = os.path.join(ROOT, "results", "sweeps")
# responsibility split used for the grids. Remark 5 says the equal split is conservative on a
# symmetric encounter, and the grid is where that shows up, so both settings are swept.
WMODE = os.environ.get("SWEEP_WMODE", "symmetric")
os.makedirs(OUT, exist_ok=True)
for _m in ("symmetric", "priority"):
    os.makedirs(os.path.join(OUT, _m), exist_ok=True)

# sine grid: separation against speed, geometry and initial separation fixed
SINE_DS = [0.4, 0.5, 0.6, 0.8, 1.0, 1.2]
SINE_V = [0.8, 1.0, 1.2, 1.3, 1.4, 1.6, 1.8, 2.0]
# circle grid: crossing angle against speed; the centre offset sets the angle
CIRC_CS = [0.6, 0.8, 1.0, 1.2, 1.4]
# the reported run flies at 1.4x the coded speeds, so this axis is relative to the reported ones
CIRC_V = [0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.8]
CIRC_V_REF = 1.4


def cell(args):
    kind, a, b, tmax = args
    # equal path distance per cell, not equal wall-clock, else fast cells get more crossings
    tmax = max(8.0, tmax / (b * (CIRC_V_REF if kind == "circles" else 1.0)))
    tag = f"_sw{WMODE[:4]}{kind}_{a}_{b}".replace(".", "p")
    js = os.path.join(OUT, f"{kind}_{a}_{b}.json".replace(".", "p", 2))
    js = os.path.join(OUT, WMODE, f"{kind}_a{a}_b{b}.json")
    tag = f"_sw{WMODE[:4]}{kind}_{a}_{b}".replace(".", "p")
    if os.path.exists(js):
        return json.load(open(js))
    cmd = [sys.executable, RUN, "--tmax", str(tmax), "--rate", "200", "--P", "100",
           "--summary-json", js, "--tag", tag,
           "--w-mode", WMODE, "--w-lead", "0.35"]
    # both grids use the initial conditions of the reported runs, one metre off path, so the cell at
    # nominal speed and nominal geometry is the run the paper reports
    if kind == "sine":
        cmd += ["--scenario", "sine", "--agents", "2", "--offpath", "--ds", str(a),
                "--vscale", str(b)]
    else:
        cmd += ["--scenario", "circles", "--agents", "4", "--offpath", "--center-scale", str(a),
                "--vscale", str(round(CIRC_V_REF * b, 4))]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not os.path.exists(js):
        print(f"  !! {kind} a={a} b={b} produced no summary\n{r.stdout[-600:]}{r.stderr[-600:]}")
        return None
    d = json.load(open(js))
    d["v_rel"] = b
    json.dump(d, open(js, "w"), indent=1)
    print(f"  {kind} a={a:<5} b={b:<5} -> {'INFEASIBLE t=%.2f' % d['t_infeasible'] if d['infeasible'] else 'ok'}"
          f"  min width={d['min_width']:.3g}")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=("sine", "circles"), required=True)
    ap.add_argument("--tmax", type=float, default=14.6)
    ap.add_argument("--jobs", type=int, default=5)
    a = ap.parse_args()
    if a.grid == "sine":
        cells = [("sine", d, v, a.tmax) for d, v in itertools.product(SINE_DS, SINE_V)]
    else:
        cells = [("circles", c, v, a.tmax) for c, v in itertools.product(CIRC_CS, CIRC_V)]
    print(f"{a.grid}: {len(cells)} cells, {a.jobs} at a time")
    with ThreadPoolExecutor(max_workers=a.jobs) as ex:
        res = [r for r in ex.map(cell, cells) if r]
    with open(os.path.join(OUT, WMODE, f"grid_{a.grid}.json"), "w") as fh:
        json.dump(res, fh, indent=1)
    ok = sum(1 for r in res if not r["infeasible"])
    print(f"done: {ok}/{len(res)} cells ran to t_max")


if __name__ == "__main__":
    main()
