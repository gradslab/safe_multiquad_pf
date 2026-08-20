r"""
V2 nominal TFL assembly (v9 Eqs. (24)-(28), Cor. 6, Lemma 7).

Given the 14-state x, a Path (paths.py), a Model (dynamics.py), gains, and desired (v_des, psi_des),
build the decoupling matrix D (Eq. 28), the drift+auxiliary vector v (Eq. 28), the nominal input
nu_TFL = D^{-1} v (Eq. 27), and the slack direction p = D^{-1} E (Eq. 40).

Rows 1-3 of D equal J_gamma(x_q) B_q with J_gamma=[Ds; dtheta] (Prop. 5); row 4 is the yaw decoupling
row. The drift entries -L_f^4 alpha_j, -L_f^4 beta1, -L_f^2 beta2 are formed by Faa di Bruno, combining
the position Lie chain L_f^k c (dynamics) with the s- and theta-derivative tensors (paths). The phase
output theta has no closed form, so this composition is necessarily numeric (Spec 0.2).
"""

import numpy as np


# ----- Faa di Bruno: total drift derivatives of a scalar field g(c(t)) up to 4th order -----
def lie_chain(g_tensors, V):
    """g_tensors = (g1,g2,g3,g4) position-derivative tensors of g (shapes (3,),(3,3),(3,3,3),(3,3,3,3)).
    V = (V1,V2,V3,V4) = L_f^{1..4} c (drift), each (3,). Returns (Lf1,Lf2,Lf3,Lf4) = L_f^{1..4}(g o c)."""
    g1, g2, g3, g4 = g_tensors
    V1, V2, V3, V4 = V
    Lf1 = g1 @ V1
    Lf2 = V1 @ g2 @ V1 + g1 @ V2
    Lf3 = (np.einsum("ijk,i,j,k->", g3, V1, V1, V1)
           + 3.0 * (V1 @ g2 @ V2) + g1 @ V3)
    Lf4 = (np.einsum("ijkl,i,j,k,l->", g4, V1, V1, V1, V1)
           + 6.0 * np.einsum("ijk,i,j,k->", g3, V1, V1, V2)
           + 3.0 * (V2 @ g2 @ V2) + 4.0 * (V1 @ g2 @ V3) + g1 @ V4)
    return Lf1, Lf2, Lf3, Lf4


E_SEL = np.array([0.0, 0.0, 1.0, 0.0])   # v9 Eq. (31): slack enters row 3 only


class TFLResult:
    __slots__ = ("D", "v", "nu_tfl", "p", "xi", "zeta", "eta", "mu", "eta2",
                 "sigma_min_D", "J_gamma", "dtheta", "Bq", "q_star", "chi", "L4_beta1")


def assemble(x, path, model, gains, v_des, psi_des, q_init):
    """Build the nominal-TFL data at state x. `gains` has keys k_xi[1..4], k_zeta[1..4],
    k_eta2/3/4, k_mu1/2. `q_init` warm-starts the projection. Returns a TFLResult."""
    x = np.asarray(x, float).reshape(14)
    y = x[6:9]                                   # position x_q
    v_q = x[9:12]

    # position Lie chain L_f^k c, k=1..4 (drift), and B_q
    V = [model.Lc(x, k) for k in (1, 2, 3, 4)]
    Bq = model.Bq(x)

    # ---- transverse channels alpha1=s1, alpha2=s2 ----
    qs = path.q_star(y, q_init)
    s_val = path.s_val(y)
    Ds = path.Ds(y); D2s = path.D2s(y); D3s = path.D3s(y); D4s = path.D4s(y)   # (2,3),(2,3,3),...
    # alpha1: lie_chain returns (L^1..L^4); xi_j = L^{j-1} alpha1, and L^4 feeds the drift row
    a1_t = (Ds[0], D2s[0], D3s[0], D4s[0])
    Lf_a1 = lie_chain(a1_t, V)
    xi = np.array([s_val[0], Lf_a1[0], Lf_a1[1], Lf_a1[2]])
    L4_a1 = Lf_a1[3]
    # alpha2
    a2_t = (Ds[1], D2s[1], D3s[1], D4s[1])
    Lf_a2 = lie_chain(a2_t, V)
    zeta = np.array([s_val[1], Lf_a2[0], Lf_a2[1], Lf_a2[2]])
    L4_a2 = Lf_a2[3]

    # ---- phase channel beta1 = theta (projected arc length) ----
    dth, D2th, D3th, D4th = path.phase_derivs(y, qs)
    b1_t = (dth, D2th, D3th, D4th)
    Lf_b1 = lie_chain(b1_t, V)
    eta = np.array([0.0, Lf_b1[0], Lf_b1[1], Lf_b1[2]])      # eta1 unused (chart dep.); eta2..4
    L4_b1 = Lf_b1[3]
    eta2 = Lf_b1[0]                                          # = dtheta . v_q = physical speed (Lemma 2)

    # ---- heading channel beta2 = psi ----
    psi = x[2]
    psidot = model.drift(x)[2]                               # L_f beta2 = psidot
    mu = np.array([psi, psidot])
    L2_b2 = model.L2_beta2_drift(x)

    # ---- decoupling matrix D (Eq. 28): rows 1-3 = [Ds;dtheta] B_q, row 4 = yaw row ----
    J_gamma = np.vstack([Ds, dth])                           # (3,3)
    D = np.zeros((4, 4))
    D[0:3, :] = J_gamma @ Bq
    D[3, :] = model.yaw_row(x)

    # ---- auxiliary inputs (Eq. 24-25) ----
    kx = gains["k_xi"]; kz = gains["k_zeta"]
    v_xi = -(kx[0] * xi[0] + kx[1] * xi[1] + kx[2] * xi[2] + kx[3] * xi[3])
    v_ze = -(kz[0] * zeta[0] + kz[1] * zeta[1] + kz[2] * zeta[2] + kz[3] * zeta[3])
    v_eta = -(gains["k_eta2"] * (eta[1] - v_des) + gains["k_eta3"] * eta[2] + gains["k_eta4"] * eta[3])
    v_mu = -(gains["k_mu1"] * (mu[0] - psi_des) + gains["k_mu2"] * mu[1])

    # ---- drift vector v (Eq. 28) ----
    v = np.array([-L4_a1 + v_xi, -L4_a2 + v_ze, -L4_b1 + v_eta, -L2_b2 + v_mu])

    # ---- nominal input and slack direction ----
    # solve D nu = v and D p = E (LU); assert conditioning
    nu_tfl = np.linalg.solve(D, v)
    p = np.linalg.solve(D, E_SEL)
    smin = np.linalg.svd(D, compute_uv=False)[-1]

    R = TFLResult()
    R.D = D; R.v = v; R.nu_tfl = nu_tfl; R.p = p
    R.xi = xi; R.zeta = zeta; R.eta = eta; R.mu = mu; R.eta2 = eta2
    R.sigma_min_D = smin; R.J_gamma = J_gamma; R.dtheta = dth; R.Bq = Bq
    R.q_star = qs; R.chi = np.concatenate([xi, zeta]); R.L4_beta1 = L4_b1
    return R
