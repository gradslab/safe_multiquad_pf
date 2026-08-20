r"""Baseline B2: the SE(3) geometric tracker of Lee, Leok and McClamroch (CDC 2010) cascaded with an
acceleration-level CBF safety filter.

This is the "well-cited nominal controller plus safety filter" architecture the paper argues against.
The point of the comparison is that the nominal controller is a *trajectory* tracker: its reference is
a function of time, so whenever the filter slows or diverts a vehicle the reference keeps running and
the vehicle is left chasing a point it can no longer reach. It leaves its path, and because the desired
attitude R_d is built from the filtered acceleration, it also loses the commanded heading.

Frame. Lee et al. write z-down (m v_dot = m g e3 - f R e3). This repository is z-up
(m v_dot = -m g e3 + f R e3), so the gravity term changes sign and the desired thrust axis is +A/||A||
rather than -A/||A||. Everything else carries over unchanged.

    A      = -k_x e_x - k_v e_v + m g e3 + m xdd_d          (z-up form of Lee eq. 12/15)
    b_3d   =  A / ||A||                                     thrust axis
    f      =  A . (R e3)                                    Lee eq. 15
    R_d    = [b_2d x b_3d, b_2d, b_3d],  b_2d = (b_3d x b_1d)/||.||
    e_R    = 1/2 (R_d^T R - R^T R_d)^v                      Lee eq. 10
    e_Om   = Om - R^T R_d Om_d                              Lee eq. 11
    M      = -k_R e_R - k_Om e_Om + Om x J Om
             - J(Om^ R^T R_d Om_d - R^T R_d Omd_d)          Lee eq. 16

Realized acceleration with exact attitude tracking is v_dot = -g e3 + A/m, so the filter acts on
u = -g e3 + A/m and maps back with A = m (u + g e3). Filtering A/m directly would be wrong by g e3.
"""
import numpy as np

_E3 = np.array([0.0, 0.0, 1.0])


def _hat(v):
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def _vee(S):
    return np.array([S[2, 1], S[0, 2], S[1, 0]])


class SE3Config:
    """Gains follow Lee et al. Sec. V, scaled by our mass; filter poles match the proposed controller."""

    def __init__(self, name, path, v_des, psi_des, q0=0.0, m=1.923,
                 k_x=16.0, k_v=5.6, k_R=8.81, k_Om=2.54,
                 ds=0.5, lam_f=(3.75, 3.75), idx=0):
        self.name = name; self.path = path; self.v_des = float(v_des)
        self.psi_des = float(psi_des); self.q0 = float(q0); self.idx = idx
        self.k_x = k_x * m; self.k_v = k_v * m; self.k_R = k_R; self.k_Om = k_Om
        self.ds = float(ds); self.lam_f = tuple(lam_f)


def reference(cfg, t):
    """Time-parameterised reference on the SAME geometric path, at the SAME physical speed.
    x_d(t) = sigma(q(t)) with arc length advancing at v_des, so |x_d'| = |v_des| exactly."""
    p = cfg.path
    q = cfg.q0
    # advance arc length by v_des*t, invert the arc-length map by Newton on a(q) = s
    target = cfg.v_des * t
    for _ in range(60):
        err = p.arclen(q, cfg.q0) - target
        sp = np.linalg.norm(p.sig(q, 1))
        if abs(err) < 1e-12 or sp < 1e-9:
            break
        q -= err / sp
    if getattr(p, "closed", False) and p.period:
        q = q % p.period
    s1 = p.sig(q, 1); s2 = p.sig(q, 2)
    sp = np.linalg.norm(s1)
    T = s1 / max(sp, 1e-9)                       # unit tangent
    xd = p.sig(q, 0)
    vd = cfg.v_des * T
    # d/dt of the unit tangent at constant arc-length speed: curvature term
    dT_dq = (s2 - T * (T @ s2)) / max(sp, 1e-9)
    ad = cfg.v_des ** 2 * dT_dq / max(sp, 1e-9)
    return xd, vd, ad, q


def _cbf_filter(i, u_nom, X12, cfgs, lam):
    """Acceleration-level CBF filter on the double integrator xdd = u.
    h = ||r||^2 - ds^2, Psi1 = hdot + lam1 h, row: Psi1dot + lam2 Psi1 >= 0, symmetric 1/2 split.
    Two or three rows in R^3: solved in closed form by active-set over a tiny problem."""
    lam1, lam2 = lam
    xi = X12[i][6:9]; vi = X12[i][9:12]
    A_rows, b_rows = [], []
    for j in range(len(X12)):
        if j == i:
            continue
        r = xi - X12[j][6:9]; rd = vi - X12[j][9:12]
        h = r @ r - cfgs[i].ds ** 2
        hd = 2.0 * (r @ rd)
        drift = 2.0 * (rd @ rd) + (lam1 + lam2) * hd + lam1 * lam2 * h
        A_rows.append(2.0 * r)                      # coefficient on u_i
        b_rows.append(0.5 * drift)                  # 2 r.u_i >= -0.5*drift
    if not A_rows:
        return u_nom, 0.0
    A = np.array(A_rows); b = np.array(b_rows)
    u = u_nom.copy()
    # projection onto the intersection of half-spaces A u >= -b (few rows; Dykstra-style sweeps)
    for _ in range(80):
        viol = A @ u + b
        k = int(np.argmin(viol))
        if viol[k] >= -1e-12:
            break
        a = A[k]; na = a @ a
        if na < 1e-12:
            break
        u = u - (viol[k] / na) * a
    return u, float(np.linalg.norm(u - u_nom))


def step_se3(i, X12, cfgs, model, t, Rd_prev, dt):
    """Return (f, M, diag) for agent i. X12: list of 12-states [phi,th,psi,p,q,r,x,y,z,vx,vy,vz]."""
    cfg = cfgs[i]; m, g = model.m, model.g
    J = np.diag([model.Ix, model.Iy, model.Iz])
    x = X12[i]
    R = model.Rot(x[0], x[1], x[2]) if hasattr(model, "Rot") else _rot(x[0], x[1], x[2])
    Om = x[3:6]
    xd, vd, ad, qref = reference(cfg, t)
    e_x = x[6:9] - xd; e_v = x[9:12] - vd

    # nominal translational command (z-up form of Lee eq. 12/15)
    A_cmd = -cfg.k_x * e_x - cfg.k_v * e_v + m * g * _E3 + m * ad
    # ---- CBF filter acts on the acceleration the tracker actually commands ----
    u_nom = -g * _E3 + A_cmd / m
    u_f, du = _cbf_filter(i, u_nom, X12, cfgs, cfg.lam_f)
    A_cmd = m * (u_f + g * _E3)

    nA = np.linalg.norm(A_cmd)
    b3d = A_cmd / max(nA, 1e-9)
    f = float(A_cmd @ (R @ _E3))                                   # Lee eq. 15
    b1d = np.array([np.cos(cfg.psi_des), np.sin(cfg.psi_des), 0.0])
    c = np.cross(b3d, b1d); nc = np.linalg.norm(c)
    if nc < 1e-6:
        b1d = np.array([0.0, 1.0, 0.0]); c = np.cross(b3d, b1d); nc = np.linalg.norm(c)
    b2d = c / nc
    Rd = np.column_stack([np.cross(b2d, b3d), b2d, b3d])

    # Om_d from finite differencing R_d (the plan permits this; step is dt)
    if Rd_prev is None:
        Om_d = np.zeros(3); Omd_d = np.zeros(3)
    else:
        Om_d = _vee(0.5 * (Rd_prev.T @ Rd - Rd.T @ Rd_prev)) / dt
        Om_d = np.clip(Om_d, -50, 50); Omd_d = np.zeros(3)

    e_R = 0.5 * _vee(Rd.T @ R - R.T @ Rd)                          # Lee eq. 10
    e_Om = Om - R.T @ Rd @ Om_d                                    # Lee eq. 11
    M = (-cfg.k_R * e_R - cfg.k_Om * e_Om + np.cross(Om, J @ Om)
         - J @ (_hat(Om) @ R.T @ Rd @ Om_d))                       # Lee eq. 16 (Omd_d = 0)
    # --- geometric path error and tangential speed, so this run is comparable to the others ---
    qs = cfg.path.q_star(x[6:9], getattr(cfg, "_qw", cfg.q0))
    cfg._qw = qs
    foot = cfg.path.sig(qs, 0)
    tan = cfg.path.sig(qs, 1); tan = tan / max(np.linalg.norm(tan), 1e-9)
    nan4 = np.full(4, np.nan)
    diag = {"name": cfg.name, "f": f, "M": M.copy(), "e_x": e_x.copy(), "e_R": e_R.copy(),
            "du": du, "nA": float(nA), "q_ref": qref, "Rd": Rd,
            # fields the logger and the comparison figures expect
            "path_err": float(np.linalg.norm(x[6:9] - foot)),      # dist(x_q, gamma)
            "traj_err": float(np.linalg.norm(e_x)),                # ||x - x_d(t)||, time-indexed
            "eta2": float(tan @ x[9:12]),
            "mu": np.array([x[2], np.nan]),                        # heading (psi_des = 0)
            "xi": nan4, "zeta": nan4, "eta": nan4,
            "nu": np.array([np.nan, M[0], M[1], M[2]]), "nu_tfl": nan4,
            "delta_star": np.nan, "delta_lo": np.nan, "delta_hi": np.nan, "width": np.nan,
            "sigma_min_D": np.nan, "sigma_min_Jgamma": np.nan, "x13_margin": f,
            "thrust_Psi1": np.nan, "n_active": 0, "feasible": True, "events": [],
            "obj": np.nan, "eq_resid": np.nan, "q_star": qs,
            "collisions": [], "attitude_h": []}
    return f, M, diag


def _rot(ph, th, ps):
    cp, sp = np.cos(ph), np.sin(ph); ct, st = np.cos(th), np.sin(th); cy, sy = np.cos(ps), np.sin(ps)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    Ry = np.array([[ct, 0, st], [0, 1, 0], [-st, 0, ct]])
    Rx = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]])
    return Rz @ Ry @ Rx
