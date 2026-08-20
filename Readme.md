# Decentralized Safe Path Following for Multiple Quadrotors on Intersecting Paths

Hamza Tariq and Adeel Akhtar, GRaDS Lab, New Jersey Institute of Technology.

**Project page: https://gradslab.github.io/safe_multiquad_pf/**

Several quadrotors each follow a fixed path in space, and the paths cross. Each vehicle must avoid
the others, stay on its own path once it is there, and hold a fixed heading. The controller is one
small quadratic program per vehicle whose four equalities are the chain assignments of transverse
feedback linearization. Two of them govern motion across the path and one governs heading. Only the
fourth, for motion along the path, is ever relaxed, by a single scalar slack. Substituting the
equalities collapses the program to one scalar, so the input and an exact feasibility test are both
available in closed form.

## What is here

```
docs/     the project page: animations, feasibility study, figures, parameter tables
sim/      the simulation code behind the paper and the page, with its own README
```

`sim/README.md` gives the commands that regenerate every result, starting with the verification gate.
The project page is built from those outputs by `sim/experiments/web/build_page.py`, and no number on
it is written by hand.

The earlier notebook implementation, `tfl_cbf_multiquad_pf.ipynb`, is kept for reference. The results
in the paper come from `sim/`.

## Citation

```bibtex
@unpublished{tariq_decentralized,
  author = {Hamza Tariq and Adeel Akhtar},
  title  = {Decentralized Safe Path Following for Multiple Quadrotors
            on Intersecting Paths},
  note   = {Under review},
  year   = {2026}
}
```
