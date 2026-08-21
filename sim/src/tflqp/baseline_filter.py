r"""
Baseline controller (Spec 2, paper Sec. V): the naive cascade the paper argues against.
Nominal TFL by matrix inversion, then a FULL-INPUT high-order-CBF safety filter (paper eq. (6)):

    nu* = argmin ||nu - nu_TFL||^2   s.t.   A(x) nu >= -b(x)     (the SAME N+4 attitude/thrust/collision
                                                                  rows as the proposed controller)

There is NO equality constraint D nu = v + E delta and NO slack: nothing pins the transverse or heading
rows, so the filter is free to move every input direction. When paths intersect it therefore leaves the
assigned path to avoid collision -- exactly the failure the proposed selective-relaxation controller avoids.
This is a genuine QP (OSQP), used for comparison experiments only. Fairness: it reuses the identical rows
(same four one-sided attitude barriers, same thrust row, same collision rows and weights) -- the ONLY
difference from the proposed controller is the dropped hard equality.
"""
import numpy as np
import cvxpy as cp

import tfl
import barriers as bar


def _solve(prob):
    """OSQP primary, CLARABEL fallback (small convex QP -> solver-invariant)."""
    try:
        prob.solve(solver=cp.OSQP, eps_abs=1e-8, eps_rel=1e-8, max_iter=40000, verbose=False)
        if prob.status not in ("optimal", "optimal_inaccurate"):
            prob.solve(solver=cp.CLARABEL, verbose=False)
    except Exception:
        prob.solve(solver=cp.CLARABEL, verbose=False)
    return prob


def solve_baseline(nu_tfl, A, b, W=None):
    """Full-input HOCBF safety filter. A:(m,4), b:(m,). Returns nu (4,) or None if infeasible."""
    W = np.eye(4) if W is None else np.asarray(W, float)
    nu = cp.Variable(4)
    obj = cp.Minimize(cp.quad_form(nu - nu_tfl, cp.psd_wrap(W)))
    cons = [A @ nu >= -b]
    prob = cp.Problem(obj, cons)
    _solve(prob)
    if nu.value is None:
        return None
    return np.asarray(nu.value).reshape(4)


def step_baseline(i, X14, kin, cfgs, model, qwarm):
    """Baseline step: same rows as controller.step, but full-input filter (no equality). Returns
    (nu or None, diag). Mirrors controller.step's diagnostics so the same figure suite applies."""
    from controller import responsibility          # reuse the (here unused) responsibility helper
    cfg = cfgs[i]; x = X14[i]; Vi, Bqi = kin[i]
    R = tfl.assemble(x, cfg.path, model, cfg.gains, cfg.v_des, cfg.psi_des, qwarm[i])
    la = cfg.lam_att
    rows = bar.attitude_rows(x, model, cfg.eps, la, la)   # paper stack: 4 attitude + (N-1) collision
    lam4 = [cfg.lam_pair] * 4
    coll = []
    for j in range(len(X14)):
        if j == i:
            continue
        Vj, _ = kin[j]
        cr = bar.collision_row(x, X14[j], Vi, Vj, Bqi, cfg.ds, lam4, cfg.w_ij)
        rows.append(cr); coll.append((j, cr))
    A = np.array([r["a"] for r in rows]); b = np.array([r["b"] for r in rows])
    nu = solve_baseline(R.nu_tfl, A, b, cfg.W)
    y = x[6:9]
    diag = {"name": cfg.name, "sigma_min_D": R.sigma_min_D, "eta2": R.eta2,
            "xi": R.xi, "zeta": R.zeta, "eta": R.eta, "mu": R.mu,
            "path_err": float(np.linalg.norm(y - cfg.path.sig(R.q_star, 0))),
            "x13": float(x[12]),
            "q_star": R.q_star, "nu_tfl": R.nu_tfl.copy(),
            "sigma_min_Jgamma": float(np.linalg.svd(R.J_gamma, compute_uv=False)[-1]),
            "collisions": [{"j": j, "dist": cr["dist"], "Psi": cr["Psi"], "c_ij": cr["c_ij"]}
                           for j, cr in coll],
            "attitude_h": [(r["name"], r["h"], r["Psi1"]) for r in rows[:4]],
            "feasible": nu is not None,
            # baseline has no slack/interval; fill NaNs so the shared logger/figures still work
            "delta_star": np.nan, "delta_lo": np.nan, "delta_hi": np.nan, "width": np.nan,
            "events": ([] if nu is not None else [{"type": "INFEASIBLE", "reason": "baseline QP"}]),
            "active": [], "n_active": np.nan}
    if nu is not None:
        diag["nu"] = nu.copy()
        # equality residual: baseline does NOT enforce D nu = v -> this is the path-leaving signature
        diag["eq_resid"] = float(np.max(np.abs((R.D @ nu - R.v)[[0, 1, 3]])))
        diag["obj"] = 0.5 * float((nu - R.nu_tfl) @ cfg.W @ (nu - R.nu_tfl))
    else:
        diag["nu"] = np.full(4, np.nan); diag["eq_resid"] = np.nan; diag["obj"] = np.nan
    return nu, diag
