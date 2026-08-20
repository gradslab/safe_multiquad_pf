r"""
V2 closed-form scalar optimizer (v9 Eqs. (40)-(45), Prop. 9). No numerical QP solver in the loop.

Reduced problem after substituting nu = nu_TFL + p*delta (Eq. 40):
  minimize  1/2 Q (delta - dhat)^2      (Q = p^T W p + P > 0)
  The MINNORM cost 1/2||nu||_W^2 + 1/2 P delta^2 makes the unconstrained minimizer
      dhat = -(p^T W nu_0)/Q                          (tex: eq:reduced_coefficients)
  and NOT zero. dhat -> 0 as P -> infinity, recovering the old nominal-deviation behaviour.
  s.t.      abar_k delta <= bbar_k ,   abar = -A p,  bbar = b + A nu_TFL      (Eq. 42)
Interval (Eq. 43): delta_lo = max{bbar_k/abar_k : abar_k<0},  delta_hi = min{bbar_k/abar_k : abar_k>0}.
Feasibility (Eq. 44): every zero-coefficient row has bbar_k>=0, and delta_lo<=delta_hi.
Optimizer: delta* = clip(dhat, delta_lo, delta_hi)          (tex: eq:projection_solution)

Zero-coefficient handling (Spec 0.4): abar_k is 'zero' when |abar_k| <= eps_a*max(1,|bbar_k|). Such a
row is feasible iff bbar_k>=0 (Eq. 44) with strict margin bbar_k>0 (Eq. 46); a violated margin is a
MARGIN_VIOLATION event. Eq. (44) failure yields an INFEASIBLE verdict with no fallback (Remark 3).
"""

import numpy as np


def reduce_and_solve(A, b, nu_tfl, p, W, P, eps_a=1e-9):
    """A: (m,4) rows a_k with a_k.nu >= -b_k; b: (m,); p=D^{-1}E; W (4x4), P>0.
    Returns dict with feasible, delta_star, nu_star, delta_lo, delta_hi, width, Q, events, active rows."""
    A = np.asarray(A, float).reshape(-1, 4)
    b = np.asarray(b, float).reshape(-1)
    Q = float(p @ W @ p + P)
    pWp = float(p @ W @ p)
    dhat = -float(p @ W @ nu_tfl) / Q
    abar = -(A @ p)                       # (m,)  # v9 Eq. (42)
    bbar = b + A @ nu_tfl
    events = []

    delta_lo, delta_hi = -np.inf, np.inf
    lo_row, hi_row = -1, -1
    feasible = True
    for k in range(A.shape[0]):
        scale = max(1.0, abs(bbar[k]))
        if abs(abar[k]) <= eps_a * scale:                 # zero-coefficient row (Eq. 44/46)
            if bbar[k] < 0.0:
                feasible = False
                events.append({"type": "INFEASIBLE", "row": k, "reason": "zero-coeff bbar<0",
                               "abar": float(abar[k]), "bbar": float(bbar[k])})
            elif bbar[k] == 0.0:
                events.append({"type": "MARGIN_VIOLATION", "row": k,
                               "abar": float(abar[k]), "bbar": float(bbar[k])})
            continue
        ratio = bbar[k] / abar[k]
        if abar[k] > 0.0:
            if ratio < delta_hi:
                delta_hi, hi_row = ratio, k
        else:
            if ratio > delta_lo:
                delta_lo, lo_row = ratio, k

    if delta_lo > delta_hi + 1e-15:
        feasible = False
        events.append({"type": "INFEASIBLE", "reason": "delta_lo>delta_hi",
                       "delta_lo": float(delta_lo), "delta_hi": float(delta_hi),
                       "lo_row": lo_row, "hi_row": hi_row})

    width = delta_hi - delta_lo
    if not feasible:
        return {"feasible": False, "delta_star": None, "nu_star": None, "delta_hat": dhat,
                "pWp": pWp, "pWp_over_Q": pWp / Q, "P_over_Q": P / Q,
                "delta_lo": delta_lo, "delta_hi": delta_hi, "width": width, "Q": Q,
                "events": events, "lo_row": lo_row, "hi_row": hi_row, "active": []}

    delta_star = max(delta_lo, min(dhat, delta_hi))        # clip(dhat,...)
    nu_star = nu_tfl + p * delta_star
    # active rows: those whose bound equals delta_star (binding)
    active = []
    if lo_row >= 0 and abs(delta_star - delta_lo) < 1e-12:
        active.append(lo_row)
    if hi_row >= 0 and abs(delta_star - delta_hi) < 1e-12:
        active.append(hi_row)
    return {"feasible": True, "delta_star": float(delta_star), "nu_star": nu_star,
            "delta_hat": dhat, "pWp": pWp, "pWp_over_Q": pWp / Q, "P_over_Q": P / Q,
            "delta_lo": delta_lo, "delta_hi": delta_hi, "width": width, "Q": Q,
            "events": events, "lo_row": lo_row, "hi_row": hi_row, "active": active}
