r"""
V2 N=2 mirrored non-planar sinusoid scenario (Spec 4 scenario 2; paper Sec. V), faithful to V1
(work/scenario_sine.py):
  quad_A: y = +5 sin(0.25 x)     quad_B: y = -5 sin(0.25 x)     z_des(x) = 1 + 0.5 sin(0.25 x)
Both travel +x; the sinusoids cross at x = 0, 4π, 8π (y=0) and the shared height field makes each a
genuine 3D intersection. d_s = 1.4 m (V1 value). A starts behind (x=-2.5) but faster than B (x=1.25),
so it catches B near a crossing -> a head-on-in-x encounter resolved in the tangential-speed channel.

Physical speeds (m/s): V1 used eta1_2 = vx*w (a scaled x-rate), vx_A=1.2, vx_B=1.0. V2's eta2 is the
projected ARC-LENGTH rate (Lemma 2), so v_des is set to the physical along-path speed ||sigma'||*vx.
Using the mean ||sigma'||~1.25 near the paths: v_des_A ~= 1.5, v_des_B ~= 1.25 m/s (documented, A7).
"""
import numpy as np
import paths as P
from controller import AgentConfig
import scenario_circles as circ    # reuse default_gains, M_HOVER

M_HOVER = circ.M_HOVER
# Compact sine (smaller amplitude, higher frequency) so the scene fits a tight zoomed camera with both
# drones + path in frame. Crossings (sin(w x)=0) at x = 0, π/w=6.28, 2π/w=12.57 ...
# Compact but SAME curvature (A·w²=0.31) as the validated A=5,w=0.25 case, so the attitude barrier is not
# over-stressed: scale A ~ 1/w². Crossings (sin(w x)=0) at x = 0, π/w=6.28, 12.57.
A_SINE, W_SINE = 1.25, 0.5
Z_AMP, Z_OMEGA, Z0 = 0.25, 0.5, 1.0
DS_SINE = 0.5           # same as the circle scenario
SIGN = {"quad_A": +1.0, "quad_B": -1.0}
V_DES = {"quad_A": 1.0, "quad_B": 0.96}          # fast + small asymmetry (the validated clean dynamic): the
#                                                  encounter is brief so A just slows to yield, no deep back-up
X0_START = {"quad_A": 2.0, "quad_B": 2.0}        # mirror start -> synchronized crossing at x=6.28 (d_s met)
NAMES = ("quad_A", "quad_B")


def make_path(name):
    return P.mirrored_sine(name, sign=SIGN[name], A=A_SINE, w=W_SINE,
                           z_amp=Z_AMP, z_omega=Z_OMEGA, z0=Z0, x_shift=0.0)


def make_configs(names=NAMES, fmin=None, lam_pair=3.75, lam_att=10.0, **kw):
    fmin = 0.05 * M_HOVER if fmin is None else fmin
    gains = circ.default_gains()                 # V1 transverse/speed/yaw gains (same as circles)
    cfgs = []
    for k, n in enumerate(names):
        cfg = AgentConfig(n, make_path(n), v_des=V_DES[n], psi_des=0.0, gains=gains, idx=k,
                          fmin=fmin, ds=DS_SINE, lam_att=lam_att, lam_pair=lam_pair,
                          lam_f=(10.0, 10.0), **kw)
        cfg.q0 = X0_START[n]                      # warm-start parameter = starting x
        cfgs.append(cfg)
    return cfgs


def initial_states(names=NAMES, offset_y=0.0, offset_z=0.0):
    """On-path (or offset) IC: at x=X0_START on the sinusoid, velocity = v_des * unit tangent, hover.
    offset_y/offset_z push it off-path to show convergence (feasible for this 2-agent scene)."""
    X = {}
    for n in names:
        pth = make_path(n); q0 = X0_START[n]
        pos = pth.sig(q0, 0).copy(); tan = pth.sig(q0, 1); tan = tan / np.linalg.norm(tan)
        x = np.zeros(14)
        x[6:9] = pos + np.array([0.0, offset_y * SIGN[n], offset_z])
        x[9:12] = V_DES[n] * tan
        x[12] = M_HOVER
        X[n] = x
    return X
