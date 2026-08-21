# Decentralized Safe Path Following for Multiple Quadrotors on Intersecting Paths

Hamza Tariq and Adeel Akhtar, GRaDS Lab, New Jersey Institute of Technology.

- **Project page** (animations, full three-controller comparison, parameters):
  https://gradslab.github.io/safe_multiquad_pf/
- **Simulation code and reproduction instructions**: [`sim/`](sim/) — start with
  [`sim/README.md`](sim/README.md), or run `bash sim/reproduce.sh` for the full
  install-to-figures pipeline.

The proposed controller reformulates transverse feedback linearization as a quadratic program
with four equality rows and one scalar slack on the along-path speed, so every collision at a
path crossing is resolved by speed alone while the path and heading stay exact. Two cascade
baselines (nominal TFL + ECBF safety filter, geometric SE(3) + the same filter) run on identical
scenarios for comparison.
