# Simulation code

Everything behind the paper and the [project page](https://gradslab.github.io/safe_multiquad_pf/).

A transverse-feedback-linearization QP whose transverse and heading rows are hard equalities and
whose tangential row carries one scalar slack. Substituting the equalities collapses the program to
a single scalar, so the controller and its feasibility test are evaluated in closed form and no
numerical solver runs in the loop.

## Layout

```
src/tflqp/        paths, dynamics, TFL assembly, barrier rows, closed-form optimizer, scenarios
experiments/      rollouts, Drake renders, manuscript figures
experiments/web/  the studies and figures on the project page
tests/            test_layer.py, the verification gate; run this first
results/          npz logs, figures, animations (not tracked)
```

## Setup

```sh
pip install -r requirements.txt
python3 tests/test_layer.py
```

The gate checks the controller algebra over several thousand random admissible states before any
scenario runs: the slack-direction lemma, every component of the slack direction, every row
authority, that the slack perturbs only the tangential row, and the minimum-norm projection. All
pass to 1e-12 or better. If it fails, nothing downstream is worth running.

**Frame note.** The paper is written z-down and this code is z-up. `B_q p = J_γ⁻¹e₃`, the collision
authority and the speed authority ±1 are frame invariant. `p₁` and the thrust and attitude
authorities change sign. The gate asserts the z-up forms.

## The reported run

The four-quadrotor result in the paper flies at 1.4 times the speeds coded in
`src/tflqp/scenario_circles.py`, which is what `--vscale 1.4` sets. Tag the outputs so the figure
scripts find them.

```sh
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 --tag _cs1p4
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 \
        --controller baseline --tag _cs1p4
python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --vscale 1.4 \
        --controller se3 --tag _cs1p4
python3 experiments/run_circles.py --scenario sine --agents 2 --offpath
python3 experiments/run_circles.py --scenario sine --agents 2 --offpath --controller baseline
python3 experiments/run_circles.py --scenario sine --agents 2 --offpath --controller se3
```

Each writes two files to `results/`: the state log, and a per-step diagnostic log carrying the
transformed states, the slack and its interval, every row authority, the barrier values, the
decoupling-matrix conditioning and the equality residual.

## The studies on the project page

```sh
python3 experiments/web/sweep.py --grid sine        # 48 rollouts, separation against speed
python3 experiments/web/sweep.py --grid circles     # 35 rollouts, crossing angle against speed
python3 experiments/web/runs.py --study all         # sampling rate, symmetric encounter, actuator box
python3 experiments/web/bench_solver.py             # projection against OSQP, 400 repetitions
python3 experiments/web/renders.py                  # 14 animations, three camera angles
```

The sweeps take roughly twenty minutes each on four cores. `renders.py` needs Drake's VTK render
engine, which is included in the pip wheel.

## Figures and the page

```sh
python3 experiments/web/figs_scenes.py              # trajectories and per-channel comparison
python3 experiments/web/figs_core.py                # slack interval, authorities, path distance
python3 experiments/web/figs_studies.py             # feasibility map, mechanism, rate, cost, splits
python3 experiments/web/page_numbers.py             # collect every quoted number
python3 experiments/web/build_page.py               # render docs/index.html
python3 experiments/web/publish.py                  # gate and copy assets into docs/
```

Every figure passes `experiments/web/webqc.py` before it is written. The gate rejects text drawn on
data, text overlapping text across panels, legends covering their own curves, labels running into a
neighbouring panel, data clipped by the view limits, and any text that would render below eleven
pixels at the width the page shows the figure. Animations are checked for blank frames, frozen
frames and cropped content. A figure that fails does not reach `docs/`.

`build_page.py` reads `results/web/numbers.json`, so no result is typed into the page by hand.

## Manuscript figures

```sh
python3 experiments/fig_track3d.py --tag _cs1p4 --paper
python3 experiments/fig_suite.py --only circles --tag-circles _cs1p4 \
        --out F1_circles_N4_channels --paper
```

`--paper` authors the canvas at the width LaTeX prints it at, so a 7 pt label on the canvas is 7 pt
on the page. Without it the figure is drawn three times too wide and every label lands near 3 pt.
Both scripts also write into the manuscript's `figures/` directory when it is present.

## Choosing P

The minimum-norm cost biases the tangential channel by `P/Q ∈ (0,1)` with `Q = pᵀWp + P`. Take `P`
large enough that `P/Q` stays near one along the path. On the four-circle scenario the bias
saturates by `P ≈ 100`: `P = 100` and `P = 1000` give the same trajectories to three digits. Any
remaining gap between `η₂` and `v_des` past that point is genuine yielding at a crossing, not cost
bias.
