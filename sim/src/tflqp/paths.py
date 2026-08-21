r"""
V2 path layer: projected arc-length phase and its derivatives (the crux, Spec 0.2).

For each agent path gamma we provide, from a raw (possibly non-unit-speed) embedding sigma_tilde(q):
  * the monotone arc-length map a(q) = \int ||sigma_tilde'|| dxi           # v9 Eq. (13)
  * the nearest raw parameter q*(y) by Newton on <sigma_tilde'(q), y-sigma_tilde(q)> = 0
  * the projected arc-length phase derivatives dtheta, D2theta, D3theta, D4theta  # v9 Eq. (9), Lemma 2
  * the transverse functions s=(s1,s2) with s^{-1}(0)=gamma and Ds..D4s        # v9 Sec. II-C

Design facts that keep this tractable and faithful to the paper:
  - The controller uses theta only through its y-derivatives (Lemma 3: eta_1 is chart-dependent and
    unused). Every derivative order of theta reduces to a CLOSED FORM in sigma_tilde',...,sigma_tilde^(5)
    and the residual w=y-sigma_tilde(q*), via implicit differentiation of the normality condition and
    the chain rule through the arc-length map A. No quadrature enters the control loop.
  - A'(q)=||sigma_tilde'(q)||, and A'',A''',A'''' are closed forms in the sigma_tilde derivatives, so the
    absolute arc length a(q) (a numeric integral) is needed only for reporting v_des and for unit tests.

Everything analytic here is validated against high-order central finite differences (Spec 0.2 / Stage B).
"""

import numpy as np
import sympy as sp


# ---------------------------------------------------------------------------
# Symbolic path definition -> fast numeric evaluators for sigma_tilde and derivs
# ---------------------------------------------------------------------------

class Path:
    """A smooth path gamma given by a raw embedding sigma_tilde(q) in R^3.

    closed=True means q is periodic with period `period` (a closed path, S^1_L); the arc-length
    map then satisfies a(q+period)=a(q)+L (Remark 1). `s_expr` (optional) is a length-2 symbolic
    transverse function of position (x,y,z); if omitted a normal-framing default is built.
    """

    def __init__(self, name, sigma_expr, q, closed=False, period=None, s_expr=None, xyz=None,
                 beta1_expr=None):
        self.name = name
        self.q = q
        self.closed = closed
        self.period = period
        # sigma_tilde and derivatives up to 5th order (5th needed for D4theta)  # v9 Prop. 5 uses up to L^4
        self._sig = [sp.Matrix(sigma_expr)]
        for _ in range(5):
            self._sig.append(sp.diff(self._sig[-1], q))
        self._sig_fns = [sp.lambdify(q, expr, "numpy") for expr in self._sig]
        # arc-length integrand ||sigma'|| and its q-derivatives A',A'',A''',A''''
        sp1 = self._sig[1]
        speed = sp.sqrt((sp1.T * sp1)[0, 0])
        self._A = [speed]
        for _ in range(4):
            self._A.append(sp.diff(self._A[-1], q))
        self._A_fns = [sp.lambdify(q, expr, "numpy") for expr in self._A]
        # transverse functions s(x,y,z) and derivatives up to 4th order  # Ds..D4s (Spec 0.2)
        self.xyz = xyz if xyz is not None else sp.symbols("x y z", real=True)
        if s_expr is None:
            s_expr = self._default_transverse()
        self.s_sym = sp.Matrix(s_expr)
        self._build_s_derivs()
        # ---- paper phase output beta1 (trimmed_proofread paper, Sec. II-C / Sec. VI) ----
        # If beta1_expr (a symbolic scalar in x,y,z) is given, the phase output is that explicit
        # chart of the nearest-point coordinate (the paper's Sec.-VI atan2 for circles) and its
        # derivative tensors come from sympy. Otherwise the phase output is the nearest-point
        # parameter q*(y) itself (the paper's varpi) via qstar_derivs. Either way beta1 is the
        # PATH-COORDINATE, not arc length: eta2 = dbeta1 . v_q is the path-coordinate rate
        # (rad/s for circles, x-rate for the sinusoids).
        self.beta1_sym = beta1_expr
        if beta1_expr is not None:
            b1 = sp.Matrix([beta1_expr])
            x_, y_, z_ = self.xyz
            self._b1_fn = sp.lambdify((x_, y_, z_), beta1_expr, "numpy")
            self._Db1_fn = sp.lambdify((x_, y_, z_), b1.jacobian(sp.Matrix([x_, y_, z_])), "numpy")
            var = [x_, y_, z_]
            self._D2b1_fn = self._make_tensor_fn(b1, var, 2)
            self._D3b1_fn = self._make_tensor_fn(b1, var, 3)
            self._D4b1_fn = self._make_tensor_fn(b1, var, 4)

    # --- sigma_tilde(q) and derivatives, as (3,) arrays ---
    def sig(self, qval, order=0):
        return np.asarray(self._sig_fns[order](float(qval)), dtype=float).reshape(3)

    def speed_derivs(self, qval):
        """[A'(q),A''(q),A'''(q),A''''(q)] = derivatives of the arc-length integrand ||sigma'||."""
        return np.array([float(f(float(qval))) for f in self._A_fns], dtype=float)

    def arclen(self, qval, q0=0.0):
        r"""a(q) = \int_{q0}^{q} ||sigma_tilde'|| dxi (Eq. 13). Numeric; used for reports/tests only."""
        from scipy.integrate import quad
        f = self._A_fns[0]
        val, _ = quad(lambda z: float(f(z)), q0, float(qval), limit=200)
        return val

    def total_length(self):
        if not self.closed:
            raise ValueError("total_length only defined for closed paths")
        return self.arclen(self.period, 0.0)

    # --- default transverse framing (unused when s_expr supplied) ---
    def _default_transverse(self):
        # Fallback: two independent normal-direction affine functions. Concrete scenarios pass s_expr.
        raise NotImplementedError("supply s_expr for concrete paths")

    def _build_s_derivs(self):
        x, y, z = self.xyz
        s = self.s_sym
        self._s_fn = sp.lambdify((x, y, z), s, "numpy")
        # Ds (2x3), and higher derivative tensors flattened via sympy derivative-by-array
        Ds = s.jacobian(sp.Matrix([x, y, z]))               # (2,3)
        self._Ds_fn = sp.lambdify((x, y, z), Ds, "numpy")
        # Second/third/fourth derivative tensors: (2,3,3), (2,3,3,3), (2,3,3,3,3)
        var = [x, y, z]
        self._D2s_fn = self._make_tensor_fn(s, var, 2)
        self._D3s_fn = self._make_tensor_fn(s, var, 3)
        self._D4s_fn = self._make_tensor_fn(s, var, 4)

    @staticmethod
    def _make_tensor_fn(s, var, order):
        import itertools
        x, y, z = var
        rows = s.rows
        comp = {}
        for r in range(rows):
            for idx in itertools.product(range(3), repeat=order):
                comp[(r,) + idx] = sp.diff(s[r, 0], *[var[k] for k in idx])
        keys = list(comp.keys())
        fns = sp.lambdify((x, y, z), [comp[k] for k in keys], "numpy")

        def tfn(xv, yv, zv, _keys=keys, _fns=fns, _rows=rows, _order=order):
            vals = _fns(float(xv), float(yv), float(zv))
            T = np.zeros((_rows,) + (3,) * _order)
            for k, v in zip(_keys, np.atleast_1d(vals)):
                T[k] = float(v)
            return T
        return tfn

    # --- transverse function value and derivative tensors at a position y ---
    def s_val(self, y):
        return np.asarray(self._s_fn(*[float(c) for c in y]), dtype=float).reshape(2)

    def Ds(self, y):
        return np.asarray(self._Ds_fn(*[float(c) for c in y]), dtype=float).reshape(2, 3)

    def D2s(self, y):
        return self._D2s_fn(*[float(c) for c in y])

    def D3s(self, y):
        return self._D3s_fn(*[float(c) for c in y])

    def D4s(self, y):
        return self._D4s_fn(*[float(c) for c in y])

    # -----------------------------------------------------------------------
    # Nearest raw parameter q*(y)  (Spec 0.2 step 1)
    # -----------------------------------------------------------------------
    def q_star(self, y, q_init, tol=1e-12, itmax=50):
        """Newton on G(q)=<sigma'(q), y-sigma(q)>=0, warm-started from q_init."""
        y = np.asarray(y, dtype=float).reshape(3)
        q = float(q_init)
        for _ in range(itmax):
            P = [self.sig(q, k) for k in range(3)]
            w = y - P[0]
            G = P[1] @ w
            Gq = P[2] @ w - P[1] @ P[1]
            if abs(Gq) < 1e-14:
                break
            step = G / Gq
            q -= step
            if abs(step) < tol:
                break
        return q

    # -----------------------------------------------------------------------
    # Projection / phase derivative tensors  (Spec 0.2 step 2; Lemma 2)
    # -----------------------------------------------------------------------
    def qstar_derivs(self, y, qstar):
        """Derivative tensors of the scalar field q*(y): (a1,a2,a3,a4) of shape (3,),(3,3),(3,3,3),(3,3,3,3).

        By implicit differentiation of the normality condition G(q,y)=<sigma'(q),y-sigma(q)>=0.
        Every term is closed form in P1..P5=sigma^(1..5)(q*) and w=y-sigma(q*). Validated against a
        quadrature-free finite difference of q_star (Stage B).
        """
        y = np.asarray(y, dtype=float).reshape(3)
        P = [self.sig(qstar, k) for k in range(6)]     # P[0..5] = sigma,...,sigma^(5)
        w = y - P[0]
        P1, P2, P3, P4, P5 = P[1], P[2], P[3], P[4], P[5]
        Gq   = P2 @ w - P1 @ P1
        Gqq  = P3 @ w - 3.0 * (P1 @ P2)
        Gqqq = P4 @ w - 4.0 * (P1 @ P3) - 3.0 * (P2 @ P2)
        Gqqqq = P5 @ w - 5.0 * (P1 @ P4) - 10.0 * (P2 @ P3)
        if abs(Gq) < 1e-14:
            raise FloatingPointError("degenerate projection (Gq~0): outside tubular neighbourhood")
        # order 1:  Gq a_i + P1[i] = 0
        a1 = -P1 / Gq
        # order 2:  a_ij = -(Gqq a_i a_j + P2[j] a_i + P2[i] a_j)/Gq
        a2 = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                a2[i, j] = -(Gqq * a1[i] * a1[j] + P2[j] * a1[i] + P2[i] * a1[j]) / Gq
        # order 3:  d/dy_k of the order-2 identity
        a3 = np.zeros((3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    Gq_k = Gqq * a1[k] + P2[k]
                    Gqq_k = Gqqq * a1[k] + P3[k]
                    term = (Gq_k * a2[i, j]
                            + Gqq_k * a1[i] * a1[j] + Gqq * (a2[i, k] * a1[j] + a1[i] * a2[j, k])
                            + P3[j] * a1[k] * a1[i] + P2[j] * a2[i, k]
                            + P3[i] * a1[k] * a1[j] + P2[i] * a2[j, k])
                    a3[i, j, k] = -term / Gq
        # order 4:  d/dy_l of the order-3 identity
        a4 = np.zeros((3, 3, 3, 3))
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    for l in range(3):
                        Gq_l = Gqq * a1[l] + P2[l]
                        Gqq_l = Gqqq * a1[l] + P3[l]
                        Gqqq_l = Gqqqq * a1[l] + P4[l]
                        Gq_k = Gqq * a1[k] + P2[k]
                        Gqq_k = Gqqq * a1[k] + P3[k]
                        d_Gq_k = Gqq_l * a1[k] + Gqq * a2[k, l] + P3[k] * a1[l]
                        d_Gqq_k = Gqqq_l * a1[k] + Gqqq * a2[k, l] + P4[k] * a1[l]
                        rest = (
                            d_Gq_k * a2[i, j] + (Gqq * a1[l] + P2[l]) * a3[i, j, k]
                            + d_Gqq_k * a1[i] * a1[j] + Gqq_k * (a2[i, l] * a1[j] + a1[i] * a2[j, l])
                            + (Gqqq * a1[l] + P3[l]) * (a2[i, k] * a1[j] + a1[i] * a2[j, k])
                            + Gqq * (a3[i, k, l] * a1[j] + a2[i, k] * a2[j, l]
                                     + a2[i, l] * a2[j, k] + a1[i] * a3[j, k, l])
                            + P4[j] * a1[l] * a1[k] * a1[i] + P3[j] * (a2[k, l] * a1[i] + a1[k] * a2[i, l])
                            + P3[j] * a1[l] * a2[i, k] + P2[j] * a3[i, k, l]
                            + P4[i] * a1[l] * a1[k] * a1[j] + P3[i] * (a2[k, l] * a1[j] + a1[k] * a2[j, l])
                            + P3[i] * a1[l] * a2[j, k] + P2[i] * a3[j, k, l]
                        )
                        a4[i, j, k, l] = -(Gq_l * a3[i, j, k] + rest) / Gq
        return a1, a2, a3, a4

    # -----------------------------------------------------------------------
    # Paper phase output beta1 (value + derivative tensors)
    # -----------------------------------------------------------------------
    def beta1_val(self, y, qstar):
        """Value of the paper's phase output beta1 at position y (qstar = warm-started projection)."""
        if self.beta1_sym is not None:
            return float(self._b1_fn(*[float(c) for c in y]))
        return float(qstar)

    def beta1_derivs(self, y, qstar):
        """(d, D2, D3, D4) tensors of beta1, shapes (3,),(3,3),(3,3,3),(3,3,3,3).

        Explicit chart (e.g. atan2 for circles): straight sympy tensors of beta1_expr.
        Default: the nearest-point parameter q*(y) via implicit differentiation (qstar_derivs)."""
        if self.beta1_sym is not None:
            yv = [float(c) for c in y]
            d1 = np.asarray(self._Db1_fn(*yv), dtype=float).reshape(3)
            d2 = self._D2b1_fn(*yv).reshape(3, 3)
            d3 = self._D3b1_fn(*yv).reshape(3, 3, 3)
            d4 = self._D4b1_fn(*yv).reshape(3, 3, 3, 3)
            return d1, d2, d3, d4
        return self.qstar_derivs(y, qstar)

    def phase_derivs(self, y, qstar):
        """Return (dtheta, D2theta, D3theta, D4theta), shapes (3,),(3,3),(3,3,3),(3,3,3,3).

        theta(y)=A(q*(y)); chain rule through the arc-length map A applied to the verified q*-tensors.
        On the path dtheta = rho^T (unit tangent), so eta_2 = dtheta . v_q is physical speed (Lemma 2).
        """
        a1, a2, a3, a4 = self.qstar_derivs(y, qstar)
        # ----- chain rule through the arc-length map theta = A(q*) -----
        Ad = self.speed_derivs(qstar)          # [A',A'',A''',A'''']
        Ap, App, Appp, Apppp = Ad[0], Ad[1], Ad[2], Ad[3]
        dth = Ap * a1
        D2th = App * np.einsum("i,j->ij", a1, a1) + Ap * a2
        D3th = (Appp * np.einsum("i,j,k->ijk", a1, a1, a1)
                + App * (np.einsum("ij,k->ijk", a2, a1) + np.einsum("ik,j->ijk", a2, a1)
                         + np.einsum("jk,i->ijk", a2, a1))
                + Ap * a3)
        # 4th order: A'''' a1^4 + A'''(6 sym a2 a1 a1) + A''(3 sym a2 a2 + 4 sym a3 a1) + A' a4
        s_a2a1a1 = (np.einsum("ij,k,l->ijkl", a2, a1, a1) + np.einsum("ik,j,l->ijkl", a2, a1, a1)
                    + np.einsum("il,j,k->ijkl", a2, a1, a1) + np.einsum("jk,i,l->ijkl", a2, a1, a1)
                    + np.einsum("jl,i,k->ijkl", a2, a1, a1) + np.einsum("kl,i,j->ijkl", a2, a1, a1))
        s_a2a2 = (np.einsum("ij,kl->ijkl", a2, a2) + np.einsum("ik,jl->ijkl", a2, a2)
                  + np.einsum("il,jk->ijkl", a2, a2))
        s_a3a1 = (np.einsum("ijk,l->ijkl", a3, a1) + np.einsum("ijl,k->ijkl", a3, a1)
                  + np.einsum("ikl,j->ijkl", a3, a1) + np.einsum("jkl,i->ijkl", a3, a1))
        D4th = (Apppp * np.einsum("i,j,k,l->ijkl", a1, a1, a1, a1)
                + Appp * s_a2a1a1 + App * (s_a2a2 + s_a3a1) + Ap * a4)
        return dth, D2th, D3th, D4th


# ---------------------------------------------------------------------------
# Concrete nominal-scenario paths
# ---------------------------------------------------------------------------

def lifted_circle(name, cx, cy, R=1.5, z_amp=0.25, z_omega=2 * np.pi / 3.0, z0=1.0):
    """Nonplanar circle: (cx+R cos q, cy+R sin q, z_des(x)), z_des(x)=z0+z_amp sin(z_omega x).

    Raw parameter q is the angle; ||sigma'|| = R sqrt(1 + z'(x)^2 sin^2 q) is phase-dependent
    (paper Sec. V), so q is NOT arc length -> the Eq. (13) conversion is mandatory.
    Transverse functions (reused from V1, Spec 0.2 permits): s1=(x-cx)^2+(y-cy)^2-R^2, s2=z-z_des(x).
    """
    q = sp.symbols("q", real=True)
    xexpr = cx + R * sp.cos(q)
    yexpr = cy + R * sp.sin(q)
    zdes = z0 + z_amp * sp.sin(z_omega * xexpr)
    sigma = [xexpr, yexpr, zdes]
    x, y, z = sp.symbols("x y z", real=True)
    zdes_xyz = z0 + z_amp * sp.sin(z_omega * x)
    s_expr = [(x - cx) ** 2 + (y - cy) ** 2 - R ** 2, z - zdes_xyz]
    # paper Sec. VI: beta1 = atan2(y-cy, x-cx), the angle chart of the nearest-point coordinate
    beta1 = sp.atan2(y - cy, x - cx)
    return Path(name, sigma, q, closed=True, period=2 * np.pi, s_expr=s_expr, xyz=(x, y, z),
                beta1_expr=beta1)


def mirrored_sine(name, sign=+1.0, A=5.0, w=0.25, z_amp=0.5, z_omega=0.25, z0=1.0, x_shift=0.0):
    """Nonplanar sinusoid: q=x, (x, sign*A sin(w(x-x_shift)), z0+z_amp sin(z_omega x)).
    ||sigma'|| = sqrt(1 + (sign*A*w cos(...))^2 + (z_amp z_omega cos(z_omega x))^2) != 1 (Sec. V).
    """
    q = sp.symbols("q", real=True)
    xexpr = q
    yexpr = sign * A * sp.sin(w * (q - x_shift))
    zdes = z0 + z_amp * sp.sin(z_omega * q)
    sigma = [xexpr, yexpr, zdes]
    x, y, z = sp.symbols("x y z", real=True)
    ypath = sign * A * sp.sin(w * (x - x_shift))
    zdes_xyz = z0 + z_amp * sp.sin(z_omega * x)
    s_expr = [y - ypath, z - zdes_xyz]
    return Path(name, sigma, q, closed=False, period=None, s_expr=s_expr, xyz=(x, y, z))
