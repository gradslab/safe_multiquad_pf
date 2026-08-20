r"""
V2 extended-quadrotor dynamics and the position-output Lie chain (Spec 0.1, 0.3; v9 Eqs. (5)-(6), Prop. 5).

State (14):  x = [phi, theta, psi, p, q, r, X, Y, Z, vx, vy, vz, x13, x14]
Input (4):   nu = [u_d, tau_p, tau_q, tau_r]         # x13=u_f (thrust), x14=u_f_dot, u_d=x14_dot

Frame: z-UP, no drag (kt=kr=0), matching the repo plant (make_env) and the paper up to the documented
e3->-e3 reflection (see docs/v2_implementation_notes.md 3.1). Everything is generated symbolically and
lambdified; the closed forms printed in the paper (B_q Eq. 22, yaw row Prop. 5(iii)) are hand-typed here
only as cross-checks (Stage B.1-2).

The controlled output is c(x)=x_q=[X,Y,Z]. We expose the drift Lie chain L_f^k c (k=0..4) and the
snap-input matrix B_q = L_g L_f^3 c (3x4). The transverse/phase channels then combine these with the
s- and theta-derivative tensors from paths.py (Faa di Bruno) in tfl.py -- the phase output theta has no
closed form, so its chain cannot be produced purely symbolically here.
"""

import numpy as np
import sympy as sp

# ------------------------------------------------------------------ symbols
_ph, _th, _ps, _p, _q, _r = sp.symbols("phi theta psi p q r", real=True)
_X, _Y, _Z, _vx, _vy, _vz = sp.symbols("X Y Z vx vy vz", real=True)
_x13, _x14 = sp.symbols("x13 x14", real=True)
_m, _g, _Ix, _Iy, _Iz = sp.symbols("m g Ix Iy Iz", positive=True)
STATE = [_ph, _th, _ps, _p, _q, _r, _X, _Y, _Z, _vx, _vy, _vz, _x13, _x14]


def _Rmat(ph, th, ps):
    """Body-to-inertial rotation, ZYX intrinsic (RollPitchYaw), matching pydrake RollPitchYaw(ph,th,ps)."""
    cph, sph = sp.cos(ph), sp.sin(ph)
    cth, sth = sp.cos(th), sp.sin(th)
    cps, sps = sp.cos(ps), sp.sin(ps)
    return sp.Matrix([
        [cps * cth, cps * sth * sph - sps * cph, cps * sth * cph + sps * sph],
        [sps * cth, sps * sth * sph + cps * cph, sps * sth * cph - cps * sph],
        [-sth,      cth * sph,                   cth * cph],
    ])


def _build_symbolic():
    ph, th, ps, p, q, r = _ph, _th, _ps, _p, _q, _r
    X, Y, Z, vx, vy, vz, x13, x14 = _X, _Y, _Z, _vx, _vy, _vz, _x13, _x14
    m, g, Ix, Iy, Iz = _m, _g, _Ix, _Iy, _Iz
    R = _Rmat(ph, th, ps)
    e3 = sp.Matrix([0, 0, 1])
    thrust_acc = (x13 / m) * (R * e3)                      # z-up: thrust along +body z
    # drift F (no drag)
    F = sp.Matrix([
        p + r * sp.cos(ph) * sp.tan(th) + q * sp.sin(ph) * sp.tan(th),
        q * sp.cos(ph) - r * sp.sin(ph),
        (q * sp.sin(ph) + r * sp.cos(ph)) / sp.cos(th),
        (Iy - Iz) * q * r / Ix,
        (Iz - Ix) * p * r / Iy,
        (Ix - Iy) * p * q / Iz,
        vx, vy, vz,
        thrust_acc[0], thrust_acc[1], thrust_acc[2] - g,
        x14,
        sp.Integer(0),
    ])
    # input columns G (14x4): u_d -> x14dot; tau -> pqr dot via 1/I
    G = sp.zeros(14, 4)
    G[13, 0] = 1
    G[3, 1] = 1 / Ix
    G[4, 2] = 1 / Iy
    G[5, 3] = 1 / Iz
    Xv = sp.Matrix(STATE)

    def Lf(hvec):
        return hvec.jacobian(Xv) * F

    c = sp.Matrix([X, Y, Z])
    Lc = [c]
    for _ in range(4):
        Lc.append(Lf(Lc[-1]))                              # L_f^0..L_f^4 c (drift)
    # B_q = d(L_f^3 c)/d nu = jac(L^3 c) wrt state times G
    L3 = Lc[3]
    Bq = L3.jacobian(Xv) * G                               # 3x4  (v9 Eq. 22)
    # yaw channel beta2 = psi ; L_f beta2 = psidot ; L_g L_f beta2 (Prop. 5(iii))
    beta2 = sp.Matrix([ps])
    Lf_beta2 = Lf(beta2)
    LgLf_beta2 = Lf_beta2.jacobian(Xv) * G                 # 1x4
    L2f_beta2 = Lf(Lf_beta2)                               # drift part of psi-ddot
    # attitude channels: for h=phi and h=theta, terms of the relative-degree-2 barrier (Eq. 33)
    att = {}
    for key, hexpr in (("phi", sp.Matrix([ph])), ("theta", sp.Matrix([th]))):
        Lfh = Lf(hexpr)                                    # h-dot (drift)
        L2fh = Lf(Lfh)                                     # drift part of h-ddot
        LgLfh = Lfh.jacobian(Xv) * G                       # 1x4
        att[key] = (Lfh, L2fh, LgLfh)
    return F, G, Lc, Bq, LgLf_beta2, L2f_beta2, att


_F, _G, _Lc, _Bq, _LgLf_beta2, _L2f_beta2, _att = _build_symbolic()

# lambdify (state + params) -> numpy
_PARAMS = [_m, _g, _Ix, _Iy, _Iz]
_ARGS = STATE + _PARAMS
_F_fn = sp.lambdify(_ARGS, _F, "numpy")
_Lc_fns = [sp.lambdify(_ARGS, Lk, "numpy") for Lk in _Lc]
_Bq_fn = sp.lambdify(_ARGS, _Bq, "numpy")
_yawrow_fn = sp.lambdify(_ARGS, _LgLf_beta2, "numpy")
_L2beta2_fn = sp.lambdify(_ARGS, _L2f_beta2, "numpy")
_R_fn = sp.lambdify([_ph, _th, _ps], _Rmat(_ph, _th, _ps), "numpy")
_att_fns = {k: (sp.lambdify(_ARGS, v[0], "numpy"), sp.lambdify(_ARGS, v[1], "numpy"),
               sp.lambdify(_ARGS, v[2], "numpy")) for k, v in _att.items()}


# ------------------------------------------------------------------ numeric API
class Model:
    """Numeric evaluator for the extended dynamics with fixed physical parameters."""

    def __init__(self, m=1.923, g=9.8, Ix=0.01152, Iy=0.01152, Iz=0.0218):
        self.p = (m, g, Ix, Iy, Iz)
        self.m, self.g, self.Ix, self.Iy, self.Iz = m, g, Ix, Iy, Iz

    def _a(self, x):
        return list(np.asarray(x, float).reshape(14)) + list(self.p)

    def R(self, x):
        return np.asarray(_R_fn(x[0], x[1], x[2]), float).reshape(3, 3)

    def drift(self, x):
        return np.asarray(_F_fn(*self._a(x)), float).reshape(14)

    def Lc(self, x, k):
        """L_f^k c (drift), k=0..4, as a (3,) array."""
        return np.asarray(_Lc_fns[k](*self._a(x)), float).reshape(3)

    def Bq(self, x):
        """Position snap-input matrix B_q = L_g L_f^3 c (3x4).  # v9 Eq. (22)"""
        return np.asarray(_Bq_fn(*self._a(x)), float).reshape(3, 4)

    def yaw_row(self, x):
        """L_g L_f beta2 (1x4), the heading decoupling row.  # v9 Prop. 5(iii)"""
        return np.asarray(_yawrow_fn(*self._a(x)), float).reshape(4)

    def L2_beta2_drift(self, x):
        """Drift part of psi-ddot = L_f^2 beta2 (scalar)."""
        return float(np.asarray(_L2beta2_fn(*self._a(x))).reshape(()))

    def att_terms(self, x, which):
        """For which in {'phi','theta'}: (h0, Lfh, L2fh_drift, LgLfh(4,)) of the attitude output.
        Used to build the relative-degree-2 one-sided barriers (Eq. 33)."""
        a = self._a(x)
        Lfh_fn, L2fh_fn, LgLfh_fn = _att_fns[which]
        h0 = x[0] if which == "phi" else x[1]
        Lfh = float(np.asarray(Lfh_fn(*a)).reshape(()))
        L2fh = float(np.asarray(L2fh_fn(*a)).reshape(()))
        LgLfh = np.asarray(LgLfh_fn(*a), float).reshape(4)
        return h0, Lfh, L2fh, LgLfh

    # ---- hand-typed closed forms (paper), for cross-check only ----
    def Bq_closed(self, x):
        """B_q = (1/m)[ R e3 | -x13 R e3^ J^{-1} ]  (z-up form of Eq. 22).
        u_d column = R e3 / m ; torque block = -(x13/m) R hat(e3) J^{-1} ; tau_r column = 0."""
        R = self.R(x)
        x13 = x[12]
        Jinv = np.diag([1 / self.Ix, 1 / self.Iy, 1 / self.Iz])
        e3hat = np.array([[0, -1.0, 0], [1.0, 0, 0], [0, 0, 0]])
        col0 = (R @ np.array([0, 0, 1.0])) / self.m
        torque = -(x13 / self.m) * (R @ e3hat @ Jinv)
        B = np.zeros((3, 4))
        B[:, 0] = col0
        B[:, 1:4] = torque
        return B

    def yaw_row_closed(self, x):
        """(0, 0, sin phi/(Iy cos th), cos phi/(Iz cos th))  # Prop. 5(iii)"""
        ph, th = x[0], x[1]
        return np.array([0.0, 0.0, np.sin(ph) / (self.Iy * np.cos(th)),
                         np.cos(ph) / (self.Iz * np.cos(th))])
