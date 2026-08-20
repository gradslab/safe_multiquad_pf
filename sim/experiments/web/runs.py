r"""Named rollouts for the project page: sampling rate, symmetric encounter, actuator box.

Each entry differs from the others in one parameter only.

    python3 experiments/web/runs.py --study rates|symmetric|actuator|all
"""
import argparse, json, os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
RUN = os.path.join(ROOT, "experiments", "run_circles.py")
SUM = os.path.join(ROOT, "results", "summaries")
os.makedirs(SUM, exist_ok=True)

CIRC = ["--scenario", "circles", "--agents", "4", "--offpath", "--P", "100"]
# the actuator ladder runs on the symmetric sinusoid encounter: it starts on path, so the input peak
# is set by the crossing rather than by a convergence transient no box would allow
SYM = ["--scenario", "sine", "--agents", "2", "--P", "100", "--rate", "200", "--tmax", "30",
       "--vdes", "1.0", "1.0", "--w-mode", "priority", "--w-lead", "0.35"]
SINE = ["--scenario", "sine", "--agents", "2", "--P", "100"]

STUDIES = {
    # does the sampled-data gap close as the period shrinks
    "rates": [(f"_rate{r}", CIRC + ["--rate", str(r), "--tmax", "30"]) for r in
              (100, 200, 500, 1000, 2000)],
    # a deliberately symmetric encounter, then the splits that break it
    "symmetric": [
        ("_sym_w50", SINE + ["--rate", "200", "--tmax", "40", "--vdes", "1.0", "1.0",
                             "--w-mode", "symmetric"]),
        ("_sym_blend", SINE + ["--rate", "200", "--tmax", "40", "--vdes", "1.0", "1.0",
                               "--w-mode", "blended", "--w-lead", "0.35", "--b-far", "8.0"]),
        ("_sym_prio", SINE + ["--rate", "200", "--tmax", "40", "--vdes", "1.0", "1.0",
                              "--w-mode", "priority", "--w-lead", "0.35"]),
    ],
    # actuator bounds as extra interval rows
    "actuator": [
        ("_box_none", SYM),
        ("_box_a", SYM + ["--tau-max", "0.030"]),
        ("_box_b", SYM + ["--tau-max", "0.026"]),
        ("_box_c", SYM + ["--tau-max", "0.023"]),
        ("_box_d", SYM + ["--tau-max", "0.021"]),
    ],
}


def launch(job):
    tag, extra = job
    js = os.path.join(SUM, f"{tag.strip('_')}.json")
    cmd = [sys.executable, RUN] + extra + ["--tag", tag, "--summary-json", js]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if not os.path.exists(js):
        print(f"  !! {tag} failed\n{r.stdout[-800:]}\n{r.stderr[-800:]}")
        return None
    d = json.load(open(js))
    print(f"  {tag:<14} {'INFEASIBLE t=%.2f' % d['t_infeasible'] if d['infeasible'] else 'ran to t_max'}"
          f"  min h={d['min_h']:+.4f}  max path err={d['max_path_err']*100:.3f} cm")
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--study", default="all")
    ap.add_argument("--jobs", type=int, default=4)
    a = ap.parse_args()
    todo = list(STUDIES) if a.study == "all" else [a.study]
    for s in todo:
        print(f"[{s}] {len(STUDIES[s])} runs")
        with ThreadPoolExecutor(max_workers=a.jobs) as ex:
            list(ex.map(launch, STUDIES[s]))


if __name__ == "__main__":
    main()
