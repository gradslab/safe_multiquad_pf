r"""
V2 nominal scenarios. Geometry/ICs reuse the V1 circle scenario (Spec 4): four nonplanar intersecting
circles, centers A(0,0) B(0,1) C(1,0) D(1,1), R=1.5, z_des(x)=1+0.25 sin(2 pi x/3), d_s=0.5.

Desired speeds are PHYSICAL (m/s), converted from V1's angular rates via the mean arc-length factor
(ambiguity A7): v_des ~ R * omega_V1. Signs preserved. Certified near-path starts: each agent begins on
its path at a chosen angle with velocity = v_des * unit tangent and hover thrust (a level-attitude start,
so a small transverse transient is expected; a fully on-manifold start is a later refinement).
"""
import numpy as np
import paths as P
from controller import AgentConfig

M_HOVER = 1.923 * 9.8

CENTERS = {"quad_A": (0.0, 0.0), "quad_B": (0.0, 1.0), "quad_C": (1.0, 0.0), "quad_D": (1.0, 1.0)}
V1_OMEGA = {"quad_A": 0.3, "quad_B": -0.25, "quad_C": 0.25, "quad_D": -0.3}
R = 1.5
# Paper (trimmed_proofread) Sec. VI: beta1 = atan2 => eta2 is the ANGULAR rate, and
# v_des = (0.3, -0.25, 0.25, -0.3) rad/s directly. Altitude field z_des(x) = 1.0 + 0.5 sin(0.25 x).
V_DES = {"quad_A": 0.45, "quad_B": -0.375, "quad_C": 0.375, "quad_D": -0.45}   # rad/s (final: 1.5x V1)
Z_AMP, Z_OMEGA, Z0 = 0.25, 2*np.pi/3, 1.0             # z_des(x) = 1.0 + 0.25 sin(2 pi x/3): two climbs and dives per circle
# Outer-corner phases: each agent starts on the side of its circle AWAY from the square centre
# (0.5,0.5), giving ~3.1 m pairwise separation at t=0 (on-path; V1's 2m off-path is infeasible w/o fallback).
Q0 = {"quad_A": 5 * np.pi / 4, "quad_B": 3 * np.pi / 4, "quad_C": 7 * np.pi / 4, "quad_D": np.pi / 4}


def default_gains(lam_s=2.0):
    """V1's PROPERLY-TUNED gains (controller_paper_ecbf.compute_tracking_terms, lines 461-467),
    mapped directly because V2 reuses V1's transverse functions (alpha1=squared radial, alpha2=altitude).
      transverse radial (alpha1):  v = -(k1 xi1 + k2 xi2 + k3 xi3 + k4 xi4),  [k1..k4]=[200,400,90,30]
      transverse altitude (alpha2):                                            [k5..k8]=[100,200,60,20]
      speed (Stage-2 triple pole -lam_s):  [k10,k11,k12] = gains_triple_pole(lam_s) = [8,12,6] at lam_s=2
      yaw:  [k13,k14]=[10,12]
    All satisfy Eq. (26) Routh-Hurwitz."""
    return dict(
        k_xi=[200.0, 400.0, 90.0, 30.0],       # [k1,k2,k3,k4]  (radial transverse chain)
        k_zeta=[100.0, 200.0, 60.0, 20.0],     # [k5,k6,k7,k8]  (altitude transverse chain)
        k_eta2=lam_s ** 3, k_eta3=3 * lam_s ** 2, k_eta4=3 * lam_s,   # speed: (s+lam_s)^3 -> [8,12,6]
        k_mu1=10.0, k_mu2=12.0,                 # [k13,k14]  (yaw)
    )


def make_configs(names=("quad_A", "quad_B", "quad_C", "quad_D"), fmin=None, lam_s=2.0,
                 lam_att=10.0, lam_pair=3.75, **kw):
    """V1-tuned defaults: lam_att=10 (V1 drake_sim), lam_pair=3.75 (V1 Stage-2, reversal-free).
    Thrust CBF: f_min=0.05*mg (a low positive-thrust floor, ~0.94 N vs hover 18.85 N) and lam_f=(10,10)
    (double pole at -10, matching the attitude pole), so the barrier keeps a ~18 N margin in normal
    flight and rarely engages (verified)."""
    fmin = 0.05 * M_HOVER if fmin is None else fmin
    gains = default_gains(lam_s=lam_s)
    cfgs = []
    for k, n in enumerate(names):
        cx, cy = CENTERS[n]
        path = P.lifted_circle(n, cx, cy, R=R, z_amp=Z_AMP, z_omega=Z_OMEGA, z0=Z0)
        v_des = V_DES[n]                            # tuned distinct physical speed (m/s)
        cfgs.append(AgentConfig(n, path, v_des=v_des, psi_des=0.0, gains=gains, idx=k,
                                fmin=fmin, ds=0.5, lam_att=lam_att, lam_pair=lam_pair,
                                lam_f=(10.0, 10.0), **kw))
        cfgs[-1].v1_omega = V1_OMEGA[n]
        cfgs[-1].q0 = Q0[n]                         # warm-start parameter (starting angle)
    return cfgs


def initial_states(names=("quad_A", "quad_B", "quad_C", "quad_D")):
    """14-state near-path IC per agent: on the path at Q0, velocity = v_des * unit tangent, level, hover."""
    X = {}
    for n in names:
        cx, cy = CENTERS[n]
        path = P.lifted_circle(n, cx, cy, R=R, z_amp=Z_AMP, z_omega=Z_OMEGA, z0=Z0)
        q0 = Q0[n]
        pos = path.sig(q0, 0)
        v_des = V_DES[n]
        vel = v_des * path.sig(q0, 1)        # qdot * dsigma/dq (v_des is the angle rate)
        x = np.zeros(14)
        x[6:9] = pos
        x[9:12] = vel
        x[12] = M_HOVER
        X[n] = x
    return X


# Legacy V1 off-path starts (each agent begins ~2 m off its circle, at rest, hover thrust) so the
# convergence to the path is visible. These are NOT on the certified set (Eq. 53) -- see Spec 4.
X0_OFFPATH = {
    "quad_A": np.array([0, 0, 0, 0, 0, 0,  3.0,  2.0, 0.2, 0, 0, 0, M_HOVER, 0], float),
    "quad_B": np.array([0, 0, 0, 0, 0, 0, -2.0, -2.0, 0.3, 0, 0, 0, M_HOVER, 0], float),
    "quad_C": np.array([0, 0, 0, 0, 0, 0, -2.0,  2.0, 0.2, 0, 0, 0, M_HOVER, 0], float),
    "quad_D": np.array([0, 0, 0, 0, 0, 0,  3.0, -1.0, 0.2, 0, 0, 0, M_HOVER, 0], float),
}


def initial_states_offpath(names=("quad_A", "quad_B", "quad_C", "quad_D"), offset_r=0.6, offset_z=-0.3):
    """Moderate off-path starts: each agent begins at its Q0 angle offset radially outward by `offset_r`
    and below by `offset_z`, at rest, hover thrust -- so convergence to the path is clearly visible while
    the transient stays feasible under gentle transverse gains. Labeled OUTSIDE_CERTIFIED_SET (Spec 4)."""
    X = {}
    for n in names:
        cx, cy = CENTERS[n]
        path = P.lifted_circle(n, cx, cy, R=R, z_amp=Z_AMP, z_omega=Z_OMEGA, z0=Z0)
        pos = path.sig(Q0[n], 0).copy()
        radial = np.array([pos[0] - cx, pos[1] - cy, 0.0])
        radial = radial / np.linalg.norm(radial)
        x = np.zeros(14)
        x[6:9] = pos + offset_r * radial + np.array([0, 0, offset_z])
        x[12] = M_HOVER
        X[n] = x
    return X
