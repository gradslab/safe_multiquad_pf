"""Verification gate for the paper controller.

Checks, on the PAPER scenarios:
  1. beta1 tensor layers vs high-order finite differences (atan2 chart on circles,
     nearest-point q* on the sinusoids).
  2. eta2 semantics: eta2 = d/dt beta1 along the drift (FD in time along the drift flow).
  3. Row stack is the paper's N+3 (4 attitude + N-1 collision), slack perturbs only the speed row,
     and the closed-form solve agrees with a numerical QP (cvxpy) on random constrained instances.
(The symbolic oracle for the assembled decoupling matrix and drift vector is tests/test_oracle.py.)
"""
import sys, os
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "src", "tflqp"))
import dynamics, tfl, controller as ctl, optimizer as opt
import scenario_circles as circ
import scenario_sine as sine

rng = np.random.default_rng(3)
mdl = dynamics.Model()
FAIL = []


def check(name, val, tol):
    ok = val < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:58s} {val:.2e} < {tol:.0e}")
    if not ok:
        FAIL.append(name)


def fd_tensors(path, y, qw, h=1e-5):
    """Finite-difference gradient and Hessian of beta1 at y (central)."""
    def b1(yy):
        qs = path.q_star(yy, qw)
        return path.beta1_val(yy, qs)
    g = np.zeros(3)
    H = np.zeros((3, 3))
    for i in range(3):
        e = np.zeros(3); e[i] = h
        g[i] = (b1(y + e) - b1(y - e)) / (2 * h)
    for i in range(3):
        for j in range(3):
            ei = np.zeros(3); ei[i] = h
            ej = np.zeros(3); ej[j] = h
            H[i, j] = (b1(y + ei + ej) - b1(y + ei - ej) - b1(y - ei + ej) + b1(y - ei - ej)) / (4 * h * h)
    return g, H


print("== 1. beta1 tensors vs finite differences ==")
worst_g, worst_H = 0.0, 0.0
cases = []
for cfgs, qk in ((circ.make_configs(), "q0"), (sine.make_configs(), "q0")):
    for cfg in cfgs[:2]:
        for _ in range(4):
            q = cfg.q0 + rng.uniform(-0.5, 0.5)
            y = cfg.path.sig(q, 0) + rng.normal(0, 0.08, 3)
            qs = cfg.path.q_star(y, q)
            d1, d2, _, _ = cfg.path.beta1_derivs(y, qs)
            g, H = fd_tensors(cfg.path, y, qs)
            worst_g = max(worst_g, np.max(np.abs(d1 - g)))
            worst_H = max(worst_H, np.max(np.abs(d2 - H)))
check("grad beta1 (atan2 + qstar) vs FD", worst_g, 5e-6)
check("hess beta1 (atan2 + qstar) vs FD", worst_H, 5e-4)

print("== 2. eta2 = d beta1/dt along drift (FD in time) ==")
worst = 0.0
for cfg in circ.make_configs()[:2] + sine.make_configs()[:1]:
    for _ in range(4):
        q = cfg.q0 + rng.uniform(-0.5, 0.5)
        x = np.zeros(14)
        x[6:9] = cfg.path.sig(q, 0) + rng.normal(0, 0.05, 3)
        x[9:12] = rng.normal(0, 0.4, 3)
        x[0:3] = rng.uniform(-0.2, 0.2, 3)
        x[3:6] = rng.normal(0, 0.3, 3)
        x[12] = circ.M_HOVER + rng.normal(0, 1.0)
        x[13] = rng.normal(0, 0.5)
        R = tfl.assemble(x, cfg.path, mdl, cfg.gains, cfg.v_des, cfg.psi_des, q)
        dt = 1e-6
        xp = x + dt * mdl.drift(x)
        qs2 = cfg.path.q_star(xp[6:9], R.q_star)
        b1p = cfg.path.beta1_val(xp[6:9], qs2)
        b1m = cfg.path.beta1_val(x[6:9], R.q_star)
        worst = max(worst, abs((b1p - b1m) / dt - R.eta2))
check("eta2 == FD d(beta1)/dt along drift", worst, 1e-4)

print("== 3. paper row stack + closed form vs numerical QP ==")
cfgs = circ.make_configs()
X0 = circ.initial_states()
X14 = [X0[c.name] for c in cfgs]
kin = [ctl.kinematics(x, mdl) for x in X14]
qw = [c.q0 for c in cfgs]
nu, diag = ctl.step(0, X14, kin, cfgs, mdl, qw)
n_expected = 4 + (len(cfgs) - 1)
check("row count == N+3 (4 attitude + N-1 collision)", abs(diag["n_rows"] - n_expected), 0.5)
check("hard-row residual (rows 1,2,4)", diag["eq_resid"], 1e-8)

try:
    import cvxpy as cp
    worst = 0.0
    for _ in range(40):
        m = rng.integers(1, 8)
        A = rng.normal(0, 1, (m, 4))
        b = rng.normal(0.5, 1, m)
        D = rng.normal(0, 1, (4, 4))
        while abs(np.linalg.det(D)) < 0.3:
            D = rng.normal(0, 1, (4, 4))
        v = rng.normal(0, 2, 4)
        nu0 = np.linalg.solve(D, v)
        p = np.linalg.solve(D, tfl.E_SEL)
        W = np.eye(4); P = 1.0
        sol = opt.reduce_and_solve(A, b, nu0, p, W, P)
        nuv = cp.Variable(4); dv = cp.Variable()
        prob = cp.Problem(cp.Minimize(0.5 * cp.sum_squares(nuv) + 0.5 * P * dv ** 2),
                          [D @ nuv == v + tfl.E_SEL * dv, A @ nuv >= -b])
        prob.solve(solver=cp.CLARABEL)
        if prob.status in ("optimal",) and sol["feasible"]:
            worst = max(worst, float(np.max(np.abs(nuv.value - sol["nu_star"]))))
        elif (prob.status not in ("optimal", "optimal_inaccurate")) != (not sol["feasible"]):
            worst = np.inf
    check("closed-form solve == cvxpy QP (40 random instances)", worst, 1e-5)
except ImportError:
    print("  [SKIP] cvxpy not available")

print()
if FAIL:
    print("*** FAILURES:", FAIL)
    sys.exit(1)
print("ALL GATE CHECKS PASS")
