r"""Closed-loop Drake rollouts for the paper's scenarios (circles N=4, sinusoids N=2).

One quaternion floating body per agent on a continuous Drake plant; the controller runs at the
--rate zero-order hold and integrates its extended thrust states internally. --controller selects
the paper's TFL-QP (proposed, closed-form solve), the TFL + safety-filter cascade (baseline), or
the SE(3) + safety-filter cascade (se3). Writes a state log and a per-step diagnostic log to
results/ (path error, transformed channels, slack and feasible interval, barrier values, equality
residual, thrust).

Usage: python3 experiments/run_circles.py --scenario circles --agents 4 --offpath --tmax 45 --rate 400
"""
import argparse
import os
import sys
import itertools

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp"))
sys.path.insert(0, PKG)

from pydrake.all import (
    DiagramBuilder, LeafSystem, AbstractValue, Simulator,
    AddMultibodyPlantSceneGraph, ExternallyAppliedSpatialForce,
    QuaternionFloatingJoint, RotationalInertia, SpatialInertia,
    SpatialForce, SpatialVelocity, RigidTransform, RotationMatrix, RollPitchYaw, Quaternion,
)

import dynamics as dyn
import controller as ctl
import scenario_circles as scn

OUT = os.path.abspath(os.path.join(HERE, "..", "results"))
os.makedirs(OUT, exist_ok=True)


def parse_agent(xfull, i, N):
    q = np.array(xfull[7 * i:7 * i + 7])
    v = np.array(xfull[7 * N + 6 * i: 7 * N + 6 * i + 6])
    quat = Quaternion(*(q[0:4] / np.linalg.norm(q[0:4])))
    Rm = RotationMatrix(quat)
    rpy = RollPitchYaw(Rm)
    pqr = Rm.matrix().T @ v[0:3]
    return np.array([rpy.roll_angle(), rpy.pitch_angle(), rpy.yaw_angle(),
                     pqr[0], pqr[1], pqr[2], q[4], q[5], q[6], v[3], v[4], v[5]]), Rm.matrix()


class V2System(LeafSystem):
    def __init__(self, cfgs, body_indices, model, dt, mode="proposed"):
        super().__init__()
        self.N = len(cfgs); self.cfgs = cfgs; self.body_indices = body_indices
        self.model = model; self.dt = dt
        self.mode = mode                        # "proposed" or "baseline" (full-input filter)
        self.qwarm = [getattr(c, "q0", 0.0) for c in cfgs]   # per-scenario path-parameter warm start
        self.infeasible = False; self.infeasible_info = None
        self.last = {}                          # last full diagnostics per agent
        self.se3 = None                          # SE(3) baseline configs, built on first use
        self.Rd_prev = [None] * self.N
        self._in = self.DeclareVectorInputPort("plant_state", 13 * self.N)
        x0 = np.tile([scn.M_HOVER, 0.0, 0.0, 0.0, 0.0], self.N)
        self.DeclareDiscreteState(x0)
        self.DeclarePeriodicDiscreteUpdateEvent(dt, 0.0, self._update)
        self.DeclareAbstractOutputPort(
            "forces", lambda: AbstractValue.Make([ExternallyAppliedSpatialForce() for _ in range(self.N)]),
            self._calc_forces)

    def _states14(self, context):
        xfull = self._in.Eval(context)
        ds = context.get_discrete_state_vector().CopyToVector().reshape(self.N, 5)
        X14 = []
        for i in range(self.N):
            s12, _ = parse_agent(xfull, i, self.N)
            X14.append(np.concatenate([s12, [ds[i, 0], ds[i, 1]]]))
        return X14, ds

    def _update(self, context, discrete_state):
        if self.infeasible:
            return
        X14, ds = self._states14(context)
        kin = [ctl.kinematics(X14[i], self.model) for i in range(self.N)]
        new = ds.copy()
        for i in range(self.N):
            if self.mode == "se3":
                import se3_baseline as s3b
                if self.se3 is None:
                    self.se3 = [s3b.SE3Config(c.name, c.path, c.v_des, c.psi_des,
                                              q0=getattr(c, "q0", 0.0), m=self.model.m,
                                              ds=c.ds, lam_f=(c.lam_pair, c.lam_pair), idx=c.idx)
                                for c in self.cfgs]
                X12 = [xx[:12] for xx in X14]
                f_cmd, M_cmd, diag = s3b.step_se3(i, X12, self.se3, self.model,
                                                  context.get_time(), self.Rd_prev[i], self.dt)
                self.Rd_prev[i] = diag["Rd"]
                # true distance-to-path for the SE(3) cascade (its own diag tracks the time
                # reference, not the path): project onto gamma with a warm-started q*
                _y = X14[i][6:9]
                _qs = self.cfgs[i].path.q_star(_y, self.qwarm[i])
                self.qwarm[i] = _qs
                diag["q_star"] = _qs
                diag["path_err"] = float(np.linalg.norm(_y - self.cfgs[i].path.sig(_qs, 0)))
                self.last[i] = diag
                new[i, 0] = f_cmd            # thrust is algebraic for SE(3): no integrators
                new[i, 1] = 0.0
                new[i, 2:5] = M_cmd
                continue
            if self.mode == "baseline":
                import baseline_filter as bfl
                nu, diag = bfl.step_baseline(i, X14, kin, self.cfgs, self.model, self.qwarm)
            else:
                nu, diag = ctl.step(i, X14, kin, self.cfgs, self.model, self.qwarm)
            self.qwarm[i] = diag["q_star"]
            self.last[i] = diag
            if nu is None:
                self.infeasible = True
                self.infeasible_info = (context.get_time(), i, diag["events"])
                return
            u_d = nu[0]; tau = nu[1:4]
            x13, x14 = ds[i, 0], ds[i, 1]
            new[i, 0] = x13 + self.dt * x14        # integrate extended thrust states (Spec 0.1)
            new[i, 1] = x14 + self.dt * u_d
            new[i, 2:5] = tau
        discrete_state.get_mutable_vector().SetFromVector(new.reshape(-1))

    def _calc_forces(self, context, output):
        xfull = self._in.Eval(context)
        ds = context.get_discrete_state_vector().CopyToVector().reshape(self.N, 5)
        forces = []
        for i in range(self.N):
            _, Rm = parse_agent(xfull, i, self.N)
            F_world = ds[i, 0] * Rm[:, 2]
            tau_world = Rm @ ds[i, 2:5]
            f = ExternallyAppliedSpatialForce()
            f.body_index = self.body_indices[i]; f.p_BoBq_B = np.zeros(3)
            f.F_Bq_W = SpatialForce(tau=tau_world, f=F_world)
            forces.append(f)
        output.set_value(forces)


def build(cfgs, model, dt, mode="proposed"):
    m, g = model.m, model.g
    J = RotationalInertia(model.Ix, model.Iy, model.Iz)
    inertia = SpatialInertia.MakeFromCentralInertia(m, np.zeros(3), J)
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -g])
    indices = []
    for c in cfgs:
        b = plant.AddRigidBody(c.name, inertia)
        plant.AddJoint(QuaternionFloatingJoint(c.name + "_j", plant.world_frame(), b.body_frame()))
        indices.append(b.index())
    plant.Finalize()
    sysv = builder.AddSystem(V2System(cfgs, indices, model, dt, mode=mode))
    builder.Connect(plant.get_state_output_port(), sysv.get_input_port(0))
    builder.Connect(sysv.get_output_port(0), plant.get_applied_spatial_force_input_port())
    return builder.Build(), plant, indices, sysv


def set_initial(plant, cfgs, X0, ctx):
    pc = plant.GetMyContextFromRoot(ctx)
    for c in cfgs:
        s = X0[c.name]
        Rm = RollPitchYaw(s[0], s[1], s[2]).ToRotationMatrix()
        body = plant.GetBodyByName(c.name)
        plant.SetFreeBodyPose(pc, body, RigidTransform(Rm, s[6:9]))
        plant.SetFreeBodySpatialVelocity(pc, body, SpatialVelocity(w=Rm.matrix() @ s[3:6], v=s[9:12]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=20.0)
    ap.add_argument("--rate", type=float, default=200.0, help="control rate Hz")
    ap.add_argument("--offpath", action="store_true", help="off-path start (show convergence)")
    ap.add_argument("--offset-r", type=float, default=1.0, help="off-path radial offset (m); 1m is feasible at N=4")
    ap.add_argument("--lam-pair", type=float, default=3.75, help="collision ECBF pole (reversal-free ~3.75)")
    ap.add_argument("--lam-att", type=float, default=10.0, help="attitude ECBF pole")
    ap.add_argument("--kt-scale", type=float, default=1.0, help="scale V1 transverse gains (gentler=<1)")
    ap.add_argument("--w-mode", choices=("symmetric", "blended", "priority"), default="symmetric",
                    help="collision responsibility: symmetric 1/2 or blended right-of-way")
    ap.add_argument("--agents", type=int, default=4, choices=(1, 2, 3, 4))
    ap.add_argument("--w-lead", type=float, default=0.45)
    ap.add_argument("--b-far", type=float, default=8.0)
    ap.add_argument("--controller", choices=("proposed", "baseline", "se3"), default="proposed")
    ap.add_argument("--P", type=float, default=100.0, help="slack weight P; large keeps P/Q near 1")
    ap.add_argument("--vmax", type=float, default=1.0, help="speed-barrier limit (m/s), |v_des|<vmax")
    ap.add_argument("--lam-v", type=float, default=20.0, help="speed-barrier triple pole")
    ap.add_argument("--vdes", type=float, nargs="*", default=None, help="override desired speeds (m/s)")
    ap.add_argument("--tag", default="", help="suffix for output filenames")
    ap.add_argument("--sine-w", type=float, default=None,
                    help="sine spatial frequency; A is rescaled as 0.3125/w^2 to hold curvature, and the "
                         "start is shifted so the approach distance to the crossing is unchanged. "
                         "Smaller w steepens the crossing angle toward head-on.")
    ap.add_argument("--ds", type=float, default=None, help="override separation distance d_s")
    ap.add_argument("--summary-json", default="", help="write a compact run summary for sweep drivers")
    ap.add_argument("--vscale", type=float, default=1.0, help="scale every desired speed")
    ap.add_argument("--center-scale", type=float, default=1.0,
                    help="scale the circle-centre offsets; larger separation = shallower crossings")
    ap.add_argument("--tau-max", type=float, default=None, help="actuator box on the three torques (N m)")
    ap.add_argument("--ud-max", type=float, default=None, help="actuator box on the thrust rate input")
    ap.add_argument("--scenario", choices=("circles", "sine", "three"), default="circles")
    args = ap.parse_args()
    dt = 1.0 / args.rate

    model = dyn.Model()
    if args.scenario == "sine":
        import scenario_sine as ssc
        names_all = ssc.NAMES[:args.agents] if args.agents <= 2 else ssc.NAMES
        if args.sine_w is not None:
            ssc.W_SINE = args.sine_w
            ssc.A_SINE = 0.3125 / args.sine_w ** 2      # hold A*w^2 (curvature) per the file's note
            _cross = np.pi / args.sine_w                 # crossing at sin(w x)=0
            ssc.X0_START = {n: _cross - 4.28 for n in ssc.NAMES}   # same approach run-up
        if args.ds is not None:
            ssc.DS_SINE = args.ds
        cfgs = ssc.make_configs(names=names_all, lam_pair=args.lam_pair, lam_att=args.lam_att,
                                w_mode=args.w_mode, w_lead=args.w_lead, b_far=args.b_far)
        X0 = ssc.initial_states(names=names_all, offset_y=(0.5 if args.offpath else 0.0),
                                offset_z=(-0.3 if args.offpath else 0.0))
        scen_tag = "sine"
    elif args.scenario == "three":
        import scenario_three as s3
        cfgs = s3.make_configs(lam_pair=args.lam_pair, lam_att=args.lam_att,
                               w_mode=args.w_mode, w_lead=args.w_lead, b_far=args.b_far)
        X0 = s3.initial_states(offpath=(1.0 if args.offpath else 0.0))
        scen_tag = "three"
    else:
        names_all = ("quad_A", "quad_B", "quad_C", "quad_D")[:args.agents]
        if args.center_scale != 1.0:
            scn.CENTERS = {n: (args.center_scale * cx, args.center_scale * cy)
                           for n, (cx, cy) in scn.CENTERS.items()}
        cfgs = scn.make_configs(names=names_all, lam_pair=args.lam_pair, lam_att=args.lam_att,
                                w_mode=args.w_mode, w_lead=args.w_lead, b_far=args.b_far)
        X0 = scn.initial_states_offpath(names=names_all, offset_r=args.offset_r) if args.offpath \
            else scn.initial_states(names=names_all)
        scen_tag = "circles"
    if args.kt_scale != 1.0:
        for c in cfgs:
            c.gains = dict(c.gains)
            c.gains["k_xi"] = [args.kt_scale * v for v in c.gains["k_xi"]]
            c.gains["k_zeta"] = [args.kt_scale * v for v in c.gains["k_zeta"]]
    for _k, _c in enumerate(cfgs):        # per-agent overrides
        _c.P = args.P; _c.v_max = args.vmax; _c.lam_v = (args.lam_v,) * 3
        if args.vdes: _c.v_des = float(args.vdes[_k % len(args.vdes)])
        if args.vscale != 1.0: _c.v_des *= args.vscale
        if args.ds is not None: _c.ds = args.ds
        _c.tau_max = args.tau_max; _c.ud_max = args.ud_max
    names = [c.name for c in cfgs]

    # t=0 separation check
    P0 = np.array([X0[n][6:9] for n in names])
    dmin0 = min((np.linalg.norm(P0[i] - P0[j])
                 for i, j in itertools.combinations(range(len(names)), 2)), default=np.inf)
    print(f"t=0 min pairwise distance = {dmin0:.3f} m (d_s={cfgs[0].ds})")

    diagram, plant, indices, sysv = build(cfgs, model, dt, mode=args.controller)
    sim = Simulator(diagram); ctx = sim.get_mutable_context()
    set_initial(plant, cfgs, X0, ctx); sim.Initialize()
    print(f"N={len(cfgs)} | rate={args.rate}Hz dt={dt:.4f} | v_des(m/s)="
          f"{[round(c.v_des,3) for c in cfgs]}")

    log = {i: [] for i in range(len(cfgs))}
    diaglog = {i: [] for i in range(len(cfgs))}
    tlog = []
    plant_ctx = plant.GetMyContextFromRoot(ctx)
    t = 0.0
    while t < args.tmax:
        t += dt
        sim.AdvanceTo(t)
        if sysv.infeasible:
            tt, ii, ev = sysv.infeasible_info
            print(f"\n*** INFEASIBLE at t={tt:.3f}s agent {names[ii]}: {ev} ***")
            break
        tlog.append(t)
        xfull = plant.get_state_output_port().Eval(plant_ctx)
        for i in range(len(cfgs)):
            s12, _ = parse_agent(xfull, i, len(cfgs))
            ds = sysv.last.get(i, {})
            log[i].append(s12.copy())
            diaglog[i].append(ds)

    tarr = np.array(tlog)
    if len(tarr) == 0:
        print("no steps completed"); return
    w = tarr > (0.5 * tarr[-1])
    print("\npost-settle diagnostics (2nd half):")
    Ss = {}
    for i, c in enumerate(cfgs):
        S = np.array(log[i]); Ss[c.name] = S
        pe = np.array([d.get("path_err", np.nan) for d in diaglog[i]])
        eta2 = np.array([d.get("eta2", np.nan) for d in diaglog[i]])
        smin = np.array([d.get("sigma_min_D", np.nan) for d in diaglog[i]])
        width = np.array([d.get("width", np.nan) for d in diaglog[i]])
        dstar = np.array([d.get("delta_star", np.nan) if d.get("delta_star") is not None else np.nan
                          for d in diaglog[i]])
        x13m = np.array([d.get("x13_margin", np.nan) for d in diaglog[i]])
        # reversal metrics (the back-and-forth the reviewer/user objected to): does eta2 cross zero
        # relative to sign(v_des), and how many sign changes of (eta2 - v_des) after settling
        sgn = np.sign(c.v_des)
        e2 = eta2[tarr > 3.0]
        if e2.size == 0 or np.all(np.isnan(e2)):
            e2 = np.array([np.nan])
            reversal = "n/a"
        else:
            reversal = "REVERSES" if (sgn * np.nanmin(sgn * e2) < -0.01) else "ok"
        de = (eta2 - c.v_des)[w]
        de = de[~np.isnan(de)]
        nsign = int(np.sum(np.abs(np.diff(np.sign(de))) > 0)) if de.size > 1 else 0
        _pw = pe[w]; _pw = _pw[~np.isnan(_pw)]
        if _pw.size == 0: _pw = np.array([np.nan])
        print(f"  {c.name}: max path_err={np.nanmax(_pw):.3e} m | eta2->v_des "
              f"{np.nanmean(eta2[w]):+.3f}/{c.v_des:+.3f} | min sigma_min(D)={np.nanmin(smin):.3e} | "
              f"max|delta*|={np.nanmax(np.abs(dstar[w])) if np.any(~np.isnan(dstar[w])) else np.nan:.2e} | "
              f"min eta2={np.nanmin(sgn*e2)*sgn:+.3f} "
              f"[{reversal}] | sign-changes={nsign}")
    # Assumption 1 (paper): collective thrust must stay positive along the closed loop
    min_x13 = np.inf
    for i in range(len(cfgs)):
        for d in diaglog[i]:
            if d:
                min_x13 = min(min_x13, d.get("x13", np.inf))
    print(f"  Assumption 1: min thrust x13 = {min_x13:.2f} N "
          f"({'holds (>0)' if min_x13 > 0 else 'VIOLATED'})")

    # collision safety
    minB = np.inf
    for i, j in itertools.combinations(range(len(cfgs)), 2):
        d2 = np.sum((Ss[names[i]][:, 6:9] - Ss[names[j]][:, 6:9]) ** 2, axis=1)
        minB = min(minB, (d2 - cfgs[0].ds ** 2).min())
    verdict = "collision-free" if minB >= 0 else ("n/a (N=1)" if len(cfgs) == 1 else "VIOLATION")
    print(f"  min pairwise barrier h = {minB:+.4f} m^2 -> {verdict}")
    print(f"  completed {len(tarr)} steps to t={tarr[-1]:.2f}s "
          f"({'INFEASIBLE' if sysv.infeasible else 'ran to tmax'})")

    ctag = {"proposed": "", "baseline": "_baseline", "se3": "_se3"}[args.controller] + args.tag
    fname = os.path.join(OUT, f"{scen_tag}_N{len(cfgs)}{ctag}.npz")
    np.savez_compressed(fname, t=tarr, **{names[i]: np.array(log[i]) for i in range(len(cfgs))})

    # comprehensive per-step diagnostics (Spec 0.5 / 5), for the figure suite
    _geo = {}
    if args.scenario == "sine":
        import scenario_sine as _ss
        _geo = {"sine_w": float(_ss.W_SINE), "sine_A": float(_ss.A_SINE)}
    diag_out = {"t": tarr, "v_des": np.array([c.v_des for c in cfgs]), "ds": cfgs[0].ds,
                **{k: np.array(v) for k, v in _geo.items()},
                "fmin": cfgs[0].fmin, "names": np.array(names)}

    def scal(i, key, default=np.nan):
        return np.array([d.get(key, default) if d else default for d in diaglog[i]], float)

    def vec(i, key, dim):
        out = np.full((len(diaglog[i]), dim), np.nan)
        for k, d in enumerate(diaglog[i]):
            if d and key in d and d[key] is not None:
                out[k] = np.asarray(d[key], float).reshape(-1)[:dim]
        return out
    for i, n in enumerate(names):
        for key in ("path_err", "eta2", "sigma_min_D", "sigma_min_Jgamma", "delta_star", "delta_lo",
                    "delta_hi", "width", "obj", "eq_resid", "x13", "n_active",
                    "delta_hat", "pWp_over_Q", "P_over_Q", "traj_err", "du"):
            diag_out[f"{n}_{key}"] = scal(i, key)
        diag_out[f"{n}_feasible"] = np.array([bool(d.get("feasible", False)) if d else False
                                              for d in diaglog[i]])
        diag_out[f"{n}_xi"] = vec(i, "xi", 4); diag_out[f"{n}_zeta"] = vec(i, "zeta", 4)
        diag_out[f"{n}_eta"] = vec(i, "eta", 4); diag_out[f"{n}_mu"] = vec(i, "mu", 2)
        diag_out[f"{n}_nu"] = vec(i, "nu", 4); diag_out[f"{n}_nu_tfl"] = vec(i, "nu_tfl", 4)
        # attitude barrier h and Psi1 (4 barriers), min collision Psi_{ij,k} over neighbours
        att_h = np.full((len(diaglog[i]), 4), np.nan); att_p = np.full((len(diaglog[i]), 4), np.nan)
        cpsi = np.full((len(diaglog[i]), 4), np.nan)
        for k, d in enumerate(diaglog[i]):
            if not d:
                continue
            if "attitude_h" in d:
                for a, (_, h, p) in enumerate(d["attitude_h"]):
                    att_h[k, a] = h; att_p[k, a] = p
            if d.get("collisions"):
                cpsi[k] = np.min([c["Psi"] for c in d["collisions"]], axis=0)
        diag_out[f"{n}_att_h"] = att_h; diag_out[f"{n}_att_Psi1"] = att_p
        for key, w in (("abar_att", 4),):
            arr = np.full((len(diaglog[i]), w), np.nan)
            for k, d in enumerate(diaglog[i]):
                if d and d.get(key) is not None: arr[k, :] = d[key]
            diag_out[f"{n}_{key}"] = arr
        nb = len(cfgs) - 1
        ac = np.full((len(diaglog[i]), nb), np.nan)
        for k, d in enumerate(diaglog[i]):
            if d and d.get("abar_coll") is not None: ac[k, :] = d["abar_coll"]
        diag_out[f"{n}_abar_coll"] = ac
        diag_out[f"{n}_coll_Psi_min"] = cpsi
    # pairwise distances (all pairs) for the min-distance / d_s figure
    for a, b in itertools.combinations(range(len(cfgs)), 2):
        dd = np.linalg.norm(Ss[names[a]][:, 6:9] - Ss[names[b]][:, 6:9], axis=1)
        diag_out[f"dist_{names[a]}_{names[b]}"] = dd
    if args.summary_json:
        import json
        wid = np.array([[d.get("width", np.nan) if d else np.nan for d in diaglog[i]]
                        for i in range(len(cfgs))], float)
        pe = np.array([[d.get("path_err", np.nan) if d else np.nan for d in diaglog[i]]
                       for i in range(len(cfgs))], float)
        wenc = tarr > 1.0                      # skip the initial transient
        summary = {
            "scenario": scen_tag, "N": len(cfgs), "rate": args.rate, "tmax": args.tmax,
            "vscale": args.vscale, "ds": float(cfgs[0].ds), "tag": args.tag,
            "sine_w": (float(__import__("scenario_sine").W_SINE) if args.scenario == "sine" else None),
            "v_des": [float(c.v_des) for c in cfgs],
            "infeasible": bool(sysv.infeasible),
            "t_end": float(tarr[-1]),
            "t_infeasible": (float(sysv.infeasible_info[0]) if sysv.infeasible else None),
            "min_width": float(np.nanmin(wid[:, wenc])) if wenc.any() else float("nan"),
            "min_h": float(minB), "d0": float(dmin0),
            "t_infeasible_scaled": (float(sysv.infeasible_info[0]) * args.vscale
                                    if sysv.infeasible else None),
            "center_scale": args.center_scale,
            "max_path_err": float(np.nanmax(pe[:, wenc])) if wenc.any() else float("nan"),
        }
        with open(args.summary_json, "w") as fh:
            json.dump(summary, fh, indent=1)
        print("wrote", args.summary_json)
    dname = os.path.join(OUT, f"{scen_tag}_N{len(cfgs)}{ctag}_diag.npz")
    np.savez_compressed(dname, **diag_out)
    print(f"wrote {fname} and {dname}")


if __name__ == "__main__":
    main()
