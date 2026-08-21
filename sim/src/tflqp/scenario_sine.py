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
# Paper (trimmed_proofread) Sec. VI: mirrored sinusoids y = +-5 sin(0.25 x), lifted onto the same
# height field z_des(x) = 1.0 + 0.5 sin(0.25 x), d_s = 0.5. beta1 is the nearest-point coordinate
# (q = x on the path), so v_des is the path-coordinate rate; the values
# (0.3, 0.25) apply. Crossings (sin(0.25 x) = 0) at x = 0, 4pi = 12.57, ...
A_SINE, W_SINE = 5.0, 0.25
Z_AMP, Z_OMEGA, Z0 = 0.25, 2*np.pi/3, 1.0   # same height field as the circles
DS_SINE = 0.8           # reported configuration
SIGN = {"quad_A": +1.0, "quad_B": -1.0}
V_DES = {"quad_A": 0.9, "quad_B": 0.75}          # desired x-rates [m/s]
X0_START = {"quad_A": -16.2, "quad_B": -13.6}
# Long run-up so BOTH agents are fully converged before the encounter:
# A crosses x=0 at 20.0 s, B at 21.7 s. The FASTER agent (A, 0.3) leads; B follows 1.7 s later and yields
# briefly (eta2 dips to ~+0.01, no reversal); closest approach grazes d_s by ~4 mm (min h ~ 0.004).
# Do NOT make the arrivals simultaneous: the exactly-symmetric configuration is the deadlock that
# strict feasibility (paper Assumption 2) excludes, and a slower-agent-leads stagger forces the
# follower to brake so hard that the attitude rows conflict with the collision row (infeasible).
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
        pos = pth.sig(q0, 0).copy()
        x = np.zeros(14)
        x[6:9] = pos + np.array([0.0, offset_y * SIGN[n], offset_z])
        x[9:12] = V_DES[n] * pth.sig(q0, 1)      # qdot * dsigma/dq (v_des is the x-rate)
        x[12] = M_HOVER
        X[n] = x
    return X
