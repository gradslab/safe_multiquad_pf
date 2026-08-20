r"""Render the animations the project page shows, one per camera angle.

The reported four-quadrotor run flies at 1.4 times the coded speeds, so those are passed explicitly
and every angle of every controller is rendered from the same run.
"""
import os, subprocess, sys
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
REN = os.path.join(ROOT, "experiments", "drake_render.py")

CIRC = ["--scenario", "circles", "--agents", "4", "--tmax", "24", "--P", "100",
        "--vdes", "0.63", "-0.525", "0.525", "-0.63"]
# framed on the first crossing: the agents travel 10 m downrange, so a camera wide enough for the
# whole run leaves the quadrotors too small to see
SINE = ["--scenario", "sine", "--agents", "2", "--tmax", "12", "--P", "100", "--pr", "0.024",
        "--qscale", "0.0026", "--tgt-x", "6.3", "--tgt-z", "2.0", "--crop-v", "0.60"]
VIEWS = {"iso": ["--view", "iso", "--zoom", "0.62"],
         "top": ["--view", "top", "--zoom", "0.88"],
         "side": ["--view", "side", "--zoom", "0.70"]}
SINE_VIEWS = {"iso": ["--view", "iso", "--zoom", "0.80"],
              "top": ["--view", "top", "--zoom", "0.95"],
              "side": ["--view", "side", "--zoom", "0.85"]}

JOBS = ([("circles", CIRC, c, v) for c in ("proposed", "baseline", "se3") for v in ("iso", "top")]
        + [("circles", CIRC, "proposed", "side")]
        + [("sine", SINE, c, v) for c in ("proposed", "baseline", "se3") for v in ("iso", "top")]
        + [("sine", SINE, "proposed", "side")])


def run(job):
    scn, base, ctrl, view = job
    tag = f"_web_{view}"
    views = SINE_VIEWS if scn == "sine" else VIEWS
    cmd = [sys.executable, REN] + base + ["--controller", ctrl] + views[view] + ["--tag", tag]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    line = [l for l in r.stdout.splitlines() if l.startswith("wrote")]
    print(f"  {scn:8s} {ctrl:9s} {view:5s} -> {line[-1][6:] if line else 'FAILED'}")
    if not line:
        print(r.stdout[-500:], r.stderr[-500:])


if __name__ == "__main__":
    only = sys.argv[1] if len(sys.argv) > 1 else ""
    jobs = [j for j in JOBS if not only or j[0] == only]
    print(f"{len(jobs)} renders")
    with ThreadPoolExecutor(max_workers=3) as ex:
        list(ex.map(run, jobs))
