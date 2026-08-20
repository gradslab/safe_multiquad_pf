r"""
V2 top-level decentralized controller (v9 Sec. IV). Per agent per step:
  1. assemble nominal TFL (D, nu_TFL, p, transformed states, sigma_min(D))          # tfl.assemble
  2. build the N+4 inequality rows: 4 attitude (Eq. 33) + 1 thrust (Eq. 34)
     + (N-1) collision (Eq. 39)                                                       # barriers.*
  3. reduce to the scalar QP and solve in closed form (Eqs. 40-45)                    # optimizer
Returns nu* (or an INFEASIBLE verdict, no fallback -- Remark 3) plus the seven paper-mandated
diagnostics (Spec 0.5): path error, eta2 (m/s), min distance + Psi_ij,k, x13-fmin + Psi_f1, the
feasible interval + width, delta* (m/s^4), sigma_min(D).

No numerical QP solver, no clipping/saturation, no controller-specific hacks (Spec 3.C).
"""

import numpy as np

import tfl
import barriers as bar
import optimizer as opt


class AgentConfig:
    def __init__(self, name, path, v_des, psi_des, gains, idx=0,
                 eps=0.1, fmin=None, ds=0.5, lam_att=1.1, lam_f=(5.0, 5.0),
                 lam_pair=3.75, W=None, P=1.0, w_ij=0.5, w_mode="symmetric", b_far=4.0, w_lead=0.4,
                 v_max=1.0, lam_v=(5.0, 5.0, 5.0)):
        self.name = name
        self.path = path
        self.idx = idx                      # agent index (priority for right-of-way)
        self.q0 = 0.0                        # initial path-parameter warm start (set by the scenario)
        self.v_des = float(v_des)
        self.psi_des = float(psi_des)
        self.gains = gains
        self.eps = eps
        self.fmin = fmin                    # set at build time (default 0.1*mg)
        self.ds = ds
        self.lam_att = lam_att
        self.lam_f = lam_f
        self.lam_pair = lam_pair            # quadruple collision pole
        self.v_max = float(v_max)           # speed limit, |v_des| < v_max  (tex Sec. IV-C)
        self.lam_v = tuple(lam_v)           # speed-barrier triple pole
        self.W = np.eye(4) if W is None else np.asarray(W, float)
        self.P = float(P)
        self.w_ij = w_ij
        self.w_mode = w_mode                # "symmetric" (w=1/2) or "blended" right-of-way
        self.b_far = b_far                  # barrier value at which right-of-way starts engaging
        self.w_lead = w_lead                # right-of-way strength: split -> (0.5-lead)/(0.5+lead)


def responsibility(cfg, i_idx, j_idx, B):
    """Collision responsibility weight w_i for pair {i,j}, with w_i + w_j = 1 preserved (Thm-5 sum).
    'symmetric': 1/2 (paper default, homogeneous agents).
    'blended' right-of-way (paper Remark 5): far apart -> ~1/2; close -> lower-index agent proceeds
    (small weight, brakes less), higher-index yields (large weight, brakes more). Antisymmetric in
    (i,j) so the pair sums to 1 by construction."""
    if cfg.w_mode == "symmetric":
        return 0.5
    if cfg.w_mode == "priority":
        # constant committed right-of-way: engages BEFORE the pair locks, so the lower-index agent
        # clears the crossing while the higher-index waits behind (breaks the co-travel limit cycle).
        lead = cfg.w_lead
    else:  # "blended": distance-modulated (gentler, but engages late)
        sigma = float(np.clip((cfg.b_far - B) / cfg.b_far, 0.0, 1.0))   # 0 far -> 1 at contact
        lead = cfg.w_lead * sigma
    return 0.5 - lead if i_idx < j_idx else 0.5 + lead


def kinematics(x, model):
    """Per-agent drift position chain [V1..V4] and B_q, shared by TFL and collision rows."""
    V = [model.Lc(x, k) for k in (1, 2, 3, 4)]
    return V, model.Bq(x)


def step(i, X14, kin, cfgs, model, qwarm, eps_a=1e-9):
    """Compute agent i's input.
    X14   : list of 14-states (all agents).
    kin   : list of (V,Bq) per agent (from kinematics()).
    cfgs  : list of AgentConfig.
    qwarm : list of warm-start q* per agent (updated in the returned diag).
    """
    cfg = cfgs[i]
    x = X14[i]
    Vi, Bqi = kin[i]

    # 1. nominal TFL
    R = tfl.assemble(x, cfg.path, model, cfg.gains, cfg.v_des, cfg.psi_des, qwarm[i])

    # 2. rows: attitude (4) + thrust (1) + speed (2) + collision (N-1)  ->  N+6 total
    rows = []
    la = cfg.lam_att
    rows += bar.attitude_rows(x, model, cfg.eps, la, la)
    rows.append(bar.thrust_row(x, cfg.fmin, cfg.lam_f[0], cfg.lam_f[1]))
    rows += bar.speed_rows(R.eta, R.L4_beta1, R.D[2, :], cfg.v_max, cfg.lam_v)
    # optional actuator box, the extension of Sec. IV-D: a bound affine in nu reduces to an interval
    # constraint of exactly the same form as every barrier row, so the closed-form test is unchanged.
    n_box = 0
    for k, lim in ((0, getattr(cfg, "ud_max", None)), (1, getattr(cfg, "tau_max", None)),
                   (2, getattr(cfg, "tau_max", None)), (3, getattr(cfg, "tau_max", None))):
        if lim is None:
            continue
        e = np.zeros(4); e[k] = 1.0
        rows.append({"a": e.copy(), "b": float(lim), "name": f"box{k}+", "h": np.nan,
                     "Psi1": np.nan})
        rows.append({"a": -e, "b": float(lim), "name": f"box{k}-", "h": np.nan, "Psi1": np.nan})
        n_box += 2

    lam4 = [cfg.lam_pair] * 4
    coll = []
    for j in range(len(X14)):
        if j == i:
            continue
        Vj, _ = kin[j]
        rij = x[6:9] - X14[j][6:9]
        B = float(rij @ rij) - cfg.ds ** 2
        w = responsibility(cfg, cfg.idx, cfgs[j].idx, B)     # blended right-of-way (w_i+w_j=1)
        cr = bar.collision_row(x, X14[j], Vi, Vj, Bqi, cfg.ds, lam4, w)
        rows.append(cr)
        coll.append((j, cr))

    A = np.array([row["a"] for row in rows])
    b = np.array([row["b"] for row in rows])

    # 3. closed-form solve
    sol = opt.reduce_and_solve(A, b, R.nu_tfl, R.p, cfg.W, cfg.P, eps_a=eps_a)

    # diagnostics (Spec 0.5 (i)-(vii))
    y = x[6:9]
    diag = {
        "name": cfg.name,
        "sigma_min_D": R.sigma_min_D,           # (vii)
        "n_rows": len(rows),
        "abar_speed": [float(-r["a"] @ R.p) for r in rows if r["name"].startswith("speed")],
        "abar_att": [float(-rows[k]["a"] @ R.p) for k in range(4)],          # roll/pitch authorities
        "abar_thrust": float(-rows[4]["a"] @ R.p),                            # = -p_1
        "abar_coll": [float(-cr["a"] @ R.p) for _, cr in coll],               # = -2<x_ij, w>
        "speed_h": [float(r["h"]) for r in rows if r["name"].startswith("speed")],
        "speed_Psi1": [float(r["Psi1"]) for r in rows if r["name"].startswith("speed")],
        "speed_Psi2": [float(r["Psi2"]) for r in rows if r["name"].startswith("speed")],
        "delta_hat": sol.get("delta_hat"), "pWp_over_Q": sol.get("pWp_over_Q"),
        "P_over_Q": sol.get("P_over_Q"),
        "eta2": R.eta2,                          # (ii) m/s
        "xi": R.xi, "zeta": R.zeta, "eta": R.eta, "mu": R.mu,
        "path_err": float(np.linalg.norm(y - cfg.path.sig(R.q_star, 0))),  # (i) dist(x_q, gamma)
        "s_val": (R.xi[0], R.zeta[0]),           # (i) transverse errors alpha1, alpha2
        "x13_margin": x[12] - cfg.fmin,          # (iv)
        "thrust_Psi1": rows[4]["Psi1"],
        "delta_star": sol.get("delta_star"),     # (vi) m/s^4
        "delta_lo": sol["delta_lo"], "delta_hi": sol["delta_hi"], "width": sol["width"],  # (v)
        "feasible": sol["feasible"],
        "events": sol["events"],
        "active": sol["active"],
        "q_star": R.q_star,
        "collisions": [{"j": j, "dist": cr["dist"], "Psi": cr["Psi"], "c_ij": cr["c_ij"]}
                       for j, cr in coll],       # (iii)
        "attitude_h": [(row["name"], row["h"], row["Psi1"]) for row in rows[:4]],
        "nu_tfl": R.nu_tfl.copy(),               # nominal TFL input
        "sigma_min_Jgamma": float(np.linalg.svd(R.J_gamma, compute_uv=False)[-1]),
        "n_active": len(sol["active"]),          # active-row count per step
        "n_box": n_box,
        "abar_box": [float(-rows[7 + k]["a"] @ R.p) for k in range(n_box)],
    }
    if sol["feasible"]:
        nu = sol["nu_star"]
        # verify hard rows exactly (Spec 0.5): D nu - v - E delta = 0
        resid = R.D @ nu - R.v - tfl.E_SEL * sol["delta_star"]
        diag["eq_resid"] = float(np.max(np.abs(resid[[0, 1, 3]])))  # rows 1,2,4 hard
        diag["obj"] = 0.5 * float((nu - R.nu_tfl) @ cfg.W @ (nu - R.nu_tfl)) + 0.5 * cfg.P * sol["delta_star"] ** 2
        diag["nu"] = nu.copy()                   # applied input
        return nu, diag
    diag["eq_resid"] = np.nan; diag["obj"] = np.nan; diag["nu"] = np.full(4, np.nan)
    return None, diag
