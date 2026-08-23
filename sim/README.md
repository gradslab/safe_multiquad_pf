# Simulation code

Everything behind the paper *"Decentralized Safe Path Following for Multiple Quadrotors on
Intersecting Paths"* and the [project page](https://gradslab.github.io/safe_multiquad_pf/).

The proposed controller reformulates transverse feedback linearization (TFL) as a quadratic
program whose transverse and heading rows are hard equalities and whose speed row carries one
scalar slack. Substituting the equalities collapses the program to a single scalar variable, so
the controller and its feasibility test are evaluated in closed form and no numerical solver runs
in the loop. Two cascade baselines run on identical scenarios: the nominal TFL controller through
an ECBF safety filter, and the geometric SE(3) controller of Lee et al. through the same filter.

## Layout

```
src/tflqp/          paths and phase layer, dynamics, TFL assembly, ECBF rows,
                    closed-form optimizer, the two scenarios, both baselines
experiments/        run_circles.py        closed-loop Drake rollouts (all three controllers)
                    analyze_paper_runs.py state-log metrics, identical for every controller
                    fig_paper_suite.py    the manuscript and project-page figures
                    drake_render.py       VTK-rendered animations (GIF), three camera views
tests/              verification gates; run these before anything else
results/            logs, figures, animations (not tracked)
```

## Reproduce everything

```sh
git clone https://github.com/gradslab/safe_multiquad_pf.git
cd safe_multiquad_pf/sim
bash reproduce.sh          # install -> gates -> 6 rollouts -> metrics -> figures
```

`reproduce.sh` runs the steps below; use them individually if you prefer.

### 1. Install

Python 3.10+ on macOS or Linux. Drake installs from pip.

```sh
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Verification gates

The gates check the controller algebra over thousands of random states, validate the phase layer
against finite differences, cross-check the closed-form solve against a numerical QP of the full
program, and compare the assembled decoupling matrix and drift vector against direct symbolic
differentiation. If a gate fails, nothing downstream is worth running.

```sh
python3 tests/test_layer.py
python3 tests/test_paper_port.py
python3 tests/test_oracle.py
```

### 3. The six rollouts

Scenario geometry, speeds, and starts live in `src/tflqp/scenario_circles.py` and
`src/tflqp/scenario_sine.py`; the commands only select scenario, controller, horizon, and rate.

```sh
for c in "" "--controller baseline" "--controller se3"; do
  python3 experiments/run_circles.py --scenario circles --agents 4 --offpath \
          --tmax 45 --rate 400 --lam-pair 3.75 --tag _paper $c
  python3 experiments/run_circles.py --scenario sine --agents 2 --offpath \
          --tmax 40 --rate 400 --ds 0.8 --tag _paper $c
done
```

Each run writes a state log and a per-step diagnostic log (path error, transformed channels, the
slack and its feasible interval, barrier values, equality residual, thrust) to `results/`.

### 4. Metrics and figures

```sh
python3 experiments/analyze_paper_runs.py --tag _paper --json results/paper_metrics.json
python3 experiments/fig_paper_suite.py    --tag _paper
```

The analysis projects every logged position onto the assigned path with a warm-started Newton
nearest-point iteration, so all three controllers are measured identically. Figures land in
`results/figs/`: the circle trajectory figure, the sinusoid trajectory figure, the circles-only
channels figure used in the paper, and the two-scenario channels figure used on the project page.

### 5. Animations (optional)

Requires Drake's VTK renderer (included in the pip install) and, for the mp4 conversion, ffmpeg.

```sh
for v in iso top side; do
  python3 experiments/drake_render.py --rate 400 --scenario circles --agents 4 --tmax 45 \
          --controller proposed --view $v --zoom 0.80 --pr 0.035 --tag _$v
  python3 experiments/drake_render.py --rate 400 --scenario sine --agents 2 --tmax 26 \
          --controller proposed --view $v --zoom 0.72 --pr 0.035 --ds 0.8 --tgt-x 0 --tag _$v
done
```

Swap `--controller` for `baseline` or `se3` for the cascades.

**Frame note.** The repository simulates z-up; the paper is written z-down. The two are related by
a reflection that leaves the controller structure unchanged, and the gates assert the z-up forms.
