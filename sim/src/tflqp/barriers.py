r"""
V2 inequality rows (v9 Eqs. (33)-(39)). Each row is returned as (a, b) meaning  a . nu >= -b,
matching Eq. (32c)  A nu >= -b. Assembled per agent into the N+4 stack:
  4 attitude rows (Eq. 33) + 1 thrust row (Eq. 34) + (N-1) collision rows (Eq. 39).

State-only admissible coordinates Psi are returned alongside for logging / the certified-set check
and the Theorem-1 barrier-invariance diagnostic (Spec 0.5).
"""

import numpy as np


# --------------------------------------------------------------------- attitude (Eq. 33)
def attitude_rows(x, model, eps, lam1, lam2):
    """Four one-sided roll/pitch barriers h_{rho, st}=(pi/2-eps)-st*rho, relative degree 2.
    Row: L_f^2 h + LgLf h . nu + (lam1+lam2) L_f h + lam1 lam2 h >= 0. Returns list of dicts with
    a,b and the state coordinates h, Psi1 (= L_f h + lam1 h)."""
    a_eps = np.pi / 2.0 - eps
    rows = []
    for which in ("phi", "theta"):
        h0, Lfh, L2fh, LgLfh = model.att_terms(x, which)
        for st in (+1.0, -1.0):
            h = a_eps - st * h0
            Lf_h = -st * Lfh
            L2f_h = -st * L2fh
            LgLf_h = -st * LgLfh
            a = LgLf_h
            b = L2f_h + (lam1 + lam2) * Lf_h + lam1 * lam2 * h
            Psi1 = Lf_h + lam1 * h
            rows.append({"a": a, "b": b, "h": h, "Psi1": Psi1,
                         "name": f"{which}_{'up' if st > 0 else 'low'}"})
    return rows


# --------------------------------------------------------------------- thrust (Eq. 34)
def thrust_row(x, fmin, lam1, lam2):
    """h_f = x13 - fmin, Psi_{f,1}=x14+lam1 h_f. Row: u_d + (lam1+lam2)x14 + lam1 lam2 (x13-fmin) >= 0."""
    x13, x14 = x[12], x[13]
    hf = x13 - fmin
    a = np.array([1.0, 0.0, 0.0, 0.0])           # coefficient on u_d only
    b = (lam1 + lam2) * x14 + lam1 * lam2 * hf
    Psi1 = x14 + lam1 * hf
    return {"a": a, "b": b, "h": hf, "Psi1": Psi1, "name": "thrust"}


# --------------------------------------------------------------------- collision (Eq. 37-39)
def _sym_elementary(lams):
    """Elementary symmetric polynomials e1..e4 of four poles (binomial HOCBF coeffs)."""
    l1, l2, l3, l4 = lams
    e1 = l1 + l2 + l3 + l4
    e2 = l1 * l2 + l1 * l3 + l1 * l4 + l2 * l3 + l2 * l4 + l3 * l4
    e3 = l1 * l2 * l3 + l1 * l2 * l4 + l1 * l3 * l4 + l2 * l3 * l4
    e4 = l1 * l2 * l3 * l4
    return e1, e2, e3, e4


def collision_row(xi, xj, Vi, Vj, Bqi, ds, lams, w_ij):
    """Pairwise collision row for agent i against neighbour j (Eq. 39).
    xi,xj: 14-states; Vi=[V1..V4]^i, Vj=[V1..V4]^j are the drift position chains L_f^k c (from tfl/dynamics);
    Bqi = B_q^i (3x4). Returns (a,b) with a.nu_i >= -b, plus Psi_{ij,0..3} for logging.

    h=||r||^2-ds^2, r=x_q^i-x_q^j. Time derivatives via Delta_k = V_k^i - V_k^j (Delta_0=r):
      h1=2 r.D1 ; h2=2 D1.D1 + 2 r.D2 ; h3=6 D1.D2 + 2 r.D3 ; h4_drift=6 D2.D2 + 8 D1.D3 + 2 r.D4.
    Input coefficient (Eq. 36): LGi Lf^3 h = 2 r^T B_q^i. c_ij = h4_drift + e1 h3 + e2 h2 + e3 h1 + e4 h."""
    r = xi[6:9] - xj[6:9]
    D1 = Vi[0] - Vj[0]; D2 = Vi[1] - Vj[1]; D3 = Vi[2] - Vj[2]; D4 = Vi[3] - Vj[3]
    h = r @ r - ds ** 2
    h1 = 2.0 * (r @ D1)
    h2 = 2.0 * (D1 @ D1) + 2.0 * (r @ D2)
    h3 = 6.0 * (D1 @ D2) + 2.0 * (r @ D3)
    h4_drift = 6.0 * (D2 @ D2) + 8.0 * (D1 @ D3) + 2.0 * (r @ D4)
    e1, e2, e3, e4 = _sym_elementary(lams)
    c_ij = h4_drift + e1 * h3 + e2 * h2 + e3 * h1 + e4 * h
    a = 2.0 * (r @ Bqi)                          # (4,)  # v9 Eq. (36)
    b = w_ij * c_ij
    # state-only admissible coordinates Psi_{ij,0..3} (Eq. 37) for logging
    l1, l2, l3, l4 = lams
    Psi0 = h
    Psi1 = h1 + l1 * h
    Psi2 = h2 + (l1 + l2) * h1 + l1 * l2 * h
    Psi3 = h3 + (l1 + l2 + l3) * h2 + (l1 * l2 + l1 * l3 + l2 * l3) * h1 + l1 * l2 * l3 * h
    return {"a": a, "b": b, "c_ij": c_ij, "Psi": (Psi0, Psi1, Psi2, Psi3),
            "dist": float(np.linalg.norm(r)), "name": "collision"}


# --------------------------------------------------------------------- speed (optional, unused)
def speed_rows(eta, L4_beta1, D_row3, v_max, lam_v):
    """Two one-sided speed barriers h_v^{i,st} = v_max - st*eta2, relative degree THREE.

    tex: the Psi definitions inlined above eq:speed_hocbf, and eq:speed_hocbf itself.
        Psi_v1^{i,st} = -st*eta3 + lam1*h
        Psi_v2^{i,st} = -st*eta4 - st*(lam1+lam2)*eta3 + lam1*lam2*h
        row: L_f Psi_v2 + L_g Psi_v2 . nu + lam3 Psi_v2 >= 0,  L_g Psi_v2 = -st * (row 3 of D)

    Reduced authority is abar = -a.p = st*(D[2,:].p) = st exactly, since D p = E. Frame-invariant.
    """
    l1, l2, l3 = lam_v
    eta2, eta3, eta4 = eta[1], eta[2], eta[3]
    rows = []
    for st in (+1.0, -1.0):
        h = v_max - st * eta2
        Psi1 = -st * eta3 + l1 * h
        Psi2 = -st * eta4 - st * (l1 + l2) * eta3 + l1 * l2 * h
        a = -st * D_row3
        b = -st * L4_beta1 - st * (l1 + l2) * eta4 - st * l1 * l2 * eta3 + l3 * Psi2
        rows.append({"a": a, "b": b, "h": h, "Psi1": Psi1, "Psi2": Psi2,
                     "name": f"speed_{'up' if st > 0 else 'low'}", "st": st})
    return rows
