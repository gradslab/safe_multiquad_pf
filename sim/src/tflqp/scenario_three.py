r"""
Symmetric three-agent intersection (Spec 4; paper Sec. V, Remark 5). Three lifted circles centred at
the vertices of an equilateral triangle (120 deg apart, distance R from the origin) with radius R, so
each circle passes through the ORIGIN -- the common 3-way intersection. Each agent starts a quarter-turn
before the origin and moves toward it; a per-agent timing OFFSET staggers the arrivals.

Varying the offset spread `s` sweeps the three Remark-5 outcomes:
  s large  -> well staggered  -> nonempty feasible interval, all traverse (liveness OK)
  s medium -> two agents meet exactly -> singleton interval with two active bounds
  s = 0    -> simultaneous symmetric approach -> opposing bounds cross -> INFEASIBLE / CBF deadlock
"""
import numpy as np
import paths as P
from controller import AgentConfig
import scenario_circles as circ

M_HOVER = circ.M_HOVER
R = 1.5
NAMES = ("quad_A", "quad_B", "quad_C")
CENTER_ANGLE = {"quad_A": np.pi / 2, "quad_B": np.pi / 2 + 2 * np.pi / 3, "quad_C": np.pi / 2 + 4 * np.pi / 3}
# CLUSTER: per-agent radial offset of the circle centre. With all offsets 0 (centre dist = R) every circle
# passes through the EXACT origin -> a true 3-into-1 point (mathematically deadlock-prone: resolving it needs
# a deep back-up). Small offsets push the closest points slightly off the origin so the three conflicts become
# PAIRWISE crossings clustered near the origin -- still visually one intersection, but each resolves with a
# gentle single dip (no deep reversal).
CLUSTER = {"quad_A": 0.0, "quad_B": 0.0, "quad_C": 0.0}   # exact common origin (cluster offsets tested,
#   all worse: near-common still forces a tight 3-way -> infeasible or deeper reversal, so keep the true point.
CENTERS = {n: ((R + CLUSTER[n]) * np.cos(a), (R + CLUSTER[n]) * np.sin(a)) for n, a in CENTER_ANGLE.items()}
# DIFFERENT physical speeds (m/s), same simultaneous start: the faster agent reaches the common origin
# first and clears while the others are still approaching, so they still MEET at the intersection but pass
# one-by-one instead of gridlocking. (Not a timing stagger -- same start phase; the ordering emerges from
# speed.) Tuned by search_three_speeds.py.
V_DES = {"quad_A": 0.58, "quad_B": 0.46, "quad_C": 0.34}   # micro-tuned: shallow yield (-0.53) & zero
#   post-encounter oscillation while all three still clear (a narrow sweet spot; wider spreads stall).
APPROACH = np.pi / 2    # start a quarter-turn before the origin
# Timing stagger (rad of phase) so the three arrivals at the common origin are SEQUENTIAL, not
# simultaneous -> each clears the intersection while the others are away (breaks the 3-into-1 CBF
# deadlock; a symmetric arrival locks all three, which right-of-way cannot resolve for 3->1).
DEFAULT_STAGGER = (0.0, 0.0, 0.0)   # symmetric: all three meet at the intersection simultaneously


def make_path(n):
    cx, cy = CENTERS[n]
    return P.lifted_circle(n, cx, cy, R=R)      # z_des(0)=1 at the origin -> genuine 3D common point


def q_origin(n):
    cx, cy = CENTERS[n]
    return np.arctan2(-cy, -cx)                 # angle on the circle that lands on the origin


def make_configs(names=NAMES, offset=None, speeds=None, lam_pair=3.75, lam_att=10.0, lam_s=2.0, **kw):
    """offset: per-agent angular timing offset (rad). lam_s: speed-loop triple pole (higher=more damped,
    shallower yield dips). Defaults to DEFAULT_STAGGER (symmetric simultaneous arrival)."""
    if offset is None:
        offset = DEFAULT_STAGGER
    speeds = dict(V_DES if speeds is None else speeds)
    fmin = 0.05 * M_HOVER
    gains = circ.default_gains(lam_s=lam_s)
    cfgs = []
    for k, n in enumerate(names):
        cfg = AgentConfig(n, make_path(n), v_des=speeds[n], psi_des=0.0, gains=gains, idx=k,
                          fmin=fmin, ds=0.5, lam_att=lam_att, lam_pair=lam_pair, lam_f=(10.0, 10.0), **kw)
        cfg.q0 = q_origin(n) - APPROACH - offset[k]     # start before the origin (+offset stagger)
        cfgs.append(cfg)
    return cfgs


def initial_states(names=NAMES, offset=None, speeds=None, offpath=1.0):
    """On/near-path IC: at the staggered approach angle, velocity = v_des * unit tangent toward the origin.
    `offpath` (m) offsets radially outward so convergence is visible (0 for exactly on-path)."""
    if offset is None:
        offset = DEFAULT_STAGGER
    speeds = dict(V_DES if speeds is None else speeds)
    X = {}
    for k, n in enumerate(names):
        cx, cy = CENTERS[n]
        pth = make_path(n); q0 = q_origin(n) - APPROACH - offset[k]
        pos = pth.sig(q0, 0).copy(); tan = pth.sig(q0, 1); tan = tan / np.linalg.norm(tan)
        radial = np.array([pos[0] - cx, pos[1] - cy, 0.0]); radial /= np.linalg.norm(radial)
        x = np.zeros(14)
        x[6:9] = pos + offpath * radial + np.array([0, 0, -0.3 if offpath else 0.0])
        x[9:12] = speeds[n] * tan
        x[12] = M_HOVER
        X[n] = x
    return X
