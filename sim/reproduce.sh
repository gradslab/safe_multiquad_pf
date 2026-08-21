#!/usr/bin/env bash
# Ground-up reproduction: install -> verification gates -> the six rollouts -> metrics -> figures.
# Run from sim/:  bash reproduce.sh
set -euo pipefail
cd "$(dirname "$0")"

echo "== 1/4 install =="
if [ ! -d .venv ]; then python3 -m venv .venv; fi
source .venv/bin/activate
pip install --quiet -r requirements.txt

echo "== 2/4 verification gates =="
python3 tests/test_layer.py
python3 tests/test_paper_port.py
python3 tests/test_oracle.py

echo "== 3/4 the six rollouts (circles N=4 and sinusoids N=2, three controllers each) =="
for c in "" "--controller baseline" "--controller se3"; do
  python3 experiments/run_circles.py --scenario circles --agents 4 --offpath \
          --tmax 45 --rate 400 --lam-pair 3.75 --tag _paper $c
  python3 experiments/run_circles.py --scenario sine --agents 2 --offpath \
          --tmax 30 --rate 400 --ds 0.8 --tag _paper $c
done

echo "== 4/4 metrics and figures =="
python3 experiments/analyze_paper_runs.py --tag _paper --json results/paper_metrics.json
python3 experiments/fig_paper_suite.py --tag _paper

echo
echo "Done. Metrics: results/paper_metrics.json   Figures: results/figs/"
echo "Animations (optional, slower): see README section 5."
