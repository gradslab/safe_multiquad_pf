"""Gate G1 oracle: symbolic cross-check of tfl.assemble on the PAPER circle outputs.

Computes D (decoupling matrix) and the drift terms L_f^4 alpha1, L_f^4 alpha2, L_f^4 beta1,
L_f^2 beta2 by DIRECT symbolic Lie differentiation of the repo's own drift/input fields
(dynamics._F, dynamics._G) applied to the paper's virtual output
    alpha1 = (x-cx)^2+(y-cy)^2-R^2,  alpha2 = z - (z0 + z_amp sin(z_omega x)),
    beta1 = atan2(y-cy, x-cx),       beta2 = psi,
then compares against tfl.assemble (which uses the Faa di Bruno lie_chain + tensor path layer,
an entirely different code path). Agreement to ~1e-6 on random states closes Gate G1.

(The V1 notebook controller cannot serve as this oracle: its alpha2 = z - const, a flat altitude,
not the paper's lifted z_des(x).)
"""
import sys, os
import numpy as np
import sympy as sp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "tflqp"))
import dynamics, tfl
import scenario_circles as circ

rng = np.random.default_rng(9)
mdl = dynamics.Model()

# symbolic drift/input from dynamics itself, with numeric params substituted
subs = dict(zip([sp.Symbol(s, positive=True) for s in ("m", "g", "Ix", "Iy", "Iz")],
                [mdl.m, mdl.g, mdl.Ix, mdl.Iy, mdl.Iz]))
F = dynamics._F.subs(subs)
G = dynamics._G.subs(subs)
Xv = sp.Matrix(dynamics.STATE)

cx, cy, R0 = 0.0, 0.0, circ.R
z0, z_amp, z_om = circ.Z0, circ.Z_AMP, circ.Z_OMEGA
_X, _Y, _Z = dynamics.STATE[6], dynamics.STATE[7], dynamics.STATE[8]
_ps = dynamics.STATE[2]

outputs = {
    "alpha1": (_X - cx) ** 2 + (_Y - cy) ** 2 - R0 ** 2,
    "alpha2": _Z - (z0 + z_amp * sp.sin(z_om * _X)),
    "beta1": sp.atan2(_Y - cy, _X - cx),
    "beta2": _ps,
}
orders = {"alpha1": 4, "alpha2": 4, "beta1": 4, "beta2": 2}

rows_D, drift_terms = {}, {}
for name, h in outputs.items():
    r = orders[name]
    hk = sp.Matrix([h])
    for _ in range(r - 1):
        hk = hk.jacobian(Xv) * F                     # L_f^{r-1} h
    LgLf = hk.jacobian(Xv) * G                       # 1x4 row of D
    Lfr = (hk.jacobian(Xv) * F)[0, 0]                # L_f^r h (drift)
    args = dynamics.STATE
    rows_D[name] = sp.lambdify(args, LgLf, "numpy")
    drift_terms[name] = sp.lambdify(args, Lfr, "numpy")

cfg = circ.make_configs()[0]           # quad_A, centered (0,0): matches cx=cy=0 above
worst_D, worst_v = 0.0, 0.0
n_ok = 0
for _ in range(25):
    q = rng.uniform(0, 2 * np.pi)
    x = np.zeros(14)
    x[6:9] = cfg.path.sig(q, 0) + rng.normal(0, 0.15, 3)
    x[9:12] = rng.normal(0, 0.5, 3)
    x[0:3] = rng.uniform(-0.35, 0.35, 3)
    x[3:6] = rng.normal(0, 0.6, 3)
    x[12] = circ.M_HOVER + rng.normal(0, 2.0)
    x[13] = rng.normal(0, 1.0)
    T = tfl.assemble(x, cfg.path, mdl, cfg.gains, cfg.v_des, cfg.psi_des, q)
    D_sym = np.vstack([np.asarray(rows_D[n](*x), float).reshape(4)
                       for n in ("alpha1", "alpha2", "beta1", "beta2")])
    worst_D = max(worst_D, np.max(np.abs(D_sym - T.D)))
    # drift entries of v: v_k = -L_f^r h_k + v_aux_k ; compare the -L_f^r part by
    # reconstructing v_aux from the transformed states exactly as tfl does
    kx, kz = cfg.gains["k_xi"], cfg.gains["k_zeta"]
    v_aux = np.array([
        -(kx[0]*T.xi[0] + kx[1]*T.xi[1] + kx[2]*T.xi[2] + kx[3]*T.xi[3]),
        -(kz[0]*T.zeta[0] + kz[1]*T.zeta[1] + kz[2]*T.zeta[2] + kz[3]*T.zeta[3]),
        -(cfg.gains["k_eta2"]*(T.eta[1]-cfg.v_des) + cfg.gains["k_eta3"]*T.eta[2]
          + cfg.gains["k_eta4"]*T.eta[3]),
        -(cfg.gains["k_mu1"]*(T.mu[0]-cfg.psi_des) + cfg.gains["k_mu2"]*T.mu[1]),
    ])
    Lfr_sym = np.array([float(np.asarray(drift_terms[n](*x)).reshape(()))
                        for n in ("alpha1", "alpha2", "beta1", "beta2")])
    v_sym = -Lfr_sym + v_aux
    worst_v = max(worst_v, np.max(np.abs(v_sym - T.v)))
    n_ok += 1

print(f"symbolic oracle over {n_ok} random states (paper circle outputs, lifted z):")
okD = worst_D < 1e-6
okv = worst_v < 1e-5
print(f"  [{'PASS' if okD else 'FAIL'}] D (assemble) == D (direct symbolic)      {worst_D:.2e} < 1e-06")
print(f"  [{'PASS' if okv else 'FAIL'}] v (assemble) == v (direct symbolic)      {worst_v:.2e} < 1e-05")
sys.exit(0 if (okD and okv) else 1)
