r"""
Render the ACTUAL Drake scene for the V2 controller: Skydio-2 quadrotor meshes + color-coded path
spheres, drawn offscreen by Drake's VTK render engine through an RGBD camera, saved as a GIF. This is
Drake drawing the quadrotors on the physics engine, not a replot of logged data.

Same V2System controller as run_circles.py (closed-form V2 QP); physical parameters from dynamics.Model
(plant == controller model). Path spheres follow the V2 nonplanar lifted circles.

Usage: python3 v2/experiments/nominal/drake_render.py --tmax 25 --fps 15
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PKG = os.path.abspath(os.path.join(HERE, "..", "src", "tflqp"))
sys.path.insert(0, PKG)
sys.path.insert(0, HERE)

from pydrake.all import (
    DiagramBuilder, Simulator, AddMultibodyPlantSceneGraph, Parser,
    QuaternionFloatingJoint, RotationalInertia, SpatialInertia,
    RigidTransform, RollPitchYaw, RotationMatrix, Sphere, Mesh,
    MakeRenderEngineVtk, RenderEngineVtkParams, RenderCameraCore, ColorRenderCamera,
    ClippingRange, CameraInfo, DepthRenderCamera, DepthRange,
)
from pydrake.systems.sensors import RgbdSensor
from PIL import Image, ImageDraw

import dynamics as dyn
import scenario_circles as scn
import run_circles as RC

RESULTS = os.path.abspath(os.path.join(HERE, "..", "results"))
RGB = {"quad_A": (0.27, 0.47, 0.67), "quad_B": (0.93, 0.40, 0.47),
       "quad_C": (0.13, 0.53, 0.20), "quad_D": (0.80, 0.73, 0.27)}

_GLTF = None


def skydio_mesh_path():
    global _GLTF
    if _GLTF is None:
        b = DiagramBuilder()
        p, _ = AddMultibodyPlantSceneGraph(b, time_step=0.0)
        parser = Parser(p)
        parser.AddModels(url="package://drake_models/skydio_2/quadrotor.urdf")
        base = os.path.join(parser.package_map().GetPath("drake_models"), "skydio_2")
        _GLTF = os.path.join(base, "skydio_2_1000_poly.gltf")
    return _GLTF


def add_quad_visual(plant, body, rgb, scale=0.0027):
    gltf = skydio_mesh_path()
    X_BG = RigidTransform(RollPitchYaw(np.pi / 2, 0.0, 0.0).ToRotationMatrix())   # gltf y-up -> body z-up
    plant.RegisterVisualGeometry(body, X_BG, Mesh(gltf, scale), "drone_mesh",
                                 np.array([[1.], [1.], [1.], [1.]]))
    beacon = np.array([[rgb[0]], [rgb[1]], [rgb[2]], [1.0]])
    plant.RegisterVisualGeometry(body, RigidTransform([0, 0, 0.075]), Sphere(0.04), "beacon", beacon)


def add_path_curve(plant, name, path, rgb, qrange=(0, 2 * np.pi), n=80, r=0.022):
    col = np.array([[rgb[0]], [rgb[1]], [rgb[2]], [1.0]])
    q0, q1 = qrange
    for k in range(n):
        q = q0 + (q1 - q0) * k / n
        pos = path.sig(q, 0)
        plant.RegisterVisualGeometry(plant.world_body(), RigidTransform(list(pos)),
                                     Sphere(r), f"path_{name}_{k}", col)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tmax", type=float, default=55.0)
    ap.add_argument("--rate", type=float, default=200.0)
    ap.add_argument("--fps", type=int, default=20)
    ap.add_argument("--capture-every", type=int, default=28, help="capture 1 of every K control steps")
    ap.add_argument("--lam-pair", type=float, default=3.75)   # V1 Stage-2 reversal-free collision pole
    ap.add_argument("--lam-att", type=float, default=10.0)
    ap.add_argument("--w-mode", choices=("symmetric", "blended"), default="symmetric")
    ap.add_argument("--w-lead", type=float, default=0.45)
    ap.add_argument("--offset-r", type=float, default=1.0, help="off-path radial start offset (m)")
    ap.add_argument("--onpath", action="store_true", help="start on the path (no convergence shown)")
    ap.add_argument("--agents", type=int, default=4, choices=(2, 3, 4))
    ap.add_argument("--scenario", choices=("circles", "sine", "three"), default="circles")
    ap.add_argument("--P", type=float, default=100.0, help="MINNORM slack weight")
    ap.add_argument("--vmax", type=float, default=1.0)
    ap.add_argument("--lam-v", type=float, default=20.0)
    ap.add_argument("--controller", choices=("proposed","baseline","se3"), default="proposed")
    ap.add_argument("--vdes", type=float, nargs="*", default=None, help="override desired speeds")
    ap.add_argument("--tag", default="", help="suffix for the output gif name")
    ap.add_argument("--sine-w", type=float, default=None,
                    help="sine spatial frequency; A rescaled as 0.3125/w^2 to hold curvature, start "
                         "shifted to keep the approach run-up. Must match run_circles.py or the GIF "
                         "will not correspond to the reported numbers.")
    ap.add_argument("--ds", type=float, default=None, help="override separation distance d_s")
    ap.add_argument("--qscale", type=float, default=0.0030, help="quadrotor mesh scale (0.0027 ~ 0.43 m)")
    ap.add_argument("--pr", type=float, default=0.055, help="path marker sphere radius [m]")
    ap.add_argument("--zoom", type=float, default=0.62, help="camera distance multiplier; <1 moves closer")
    ap.add_argument("--view", choices=("iso", "top", "side", "front"), default="iso",
                    help="camera preset; iso keeps the per-scenario default")
    ap.add_argument("--tgt-z", type=float, default=None, help="camera target height [m]")
    ap.add_argument("--crop-v", type=float, default=1.0,
                    help="keep this fraction of the frame height, centred; a wide flat scene wastes "
                         "most of a 4:3 frame on empty sky")
    ap.add_argument("--tgt-x", type=float, default=None, help="camera target x [m]; sine travels "
                    "downrange, so the default crossing-centred target loses the agents late in a run")
    ap.add_argument("--elev", type=float, default=None, help="camera elevation [deg], overrides --view")
    ap.add_argument("--azim", type=float, default=None, help="camera azimuth [deg], overrides --view")
    ap.add_argument("--w", type=int, default=900); ap.add_argument("--h", type=int, default=680)
    args = ap.parse_args()
    dt = 1.0 / args.rate

    model = dyn.Model()
    if args.scenario == "sine":
        import scenario_sine as ssc
        names_sel = ssc.NAMES
        if args.sine_w is not None:
            ssc.W_SINE = args.sine_w
            ssc.A_SINE = 0.3125 / args.sine_w ** 2
            _cross = np.pi / args.sine_w
            ssc.X0_START = {n: _cross - 4.28 for n in ssc.NAMES}
        if args.ds is not None:
            ssc.DS_SINE = args.ds
        cfgs = ssc.make_configs(names=names_sel, lam_pair=args.lam_pair, lam_att=args.lam_att,
                                w_mode=args.w_mode, b_far=8.0, w_lead=args.w_lead)
        # 1 m off-path start so convergence is visible (offset in y ~= 1 m path distance for the gentle sine)
        X0 = ssc.initial_states(names=names_sel, offset_y=(0.0 if args.onpath else 1.0),
                                offset_z=(0.0 if args.onpath else -0.3))
    elif args.scenario == "three":
        import scenario_three as s3
        names_sel = s3.NAMES
        cfgs = s3.make_configs(names=names_sel, lam_pair=args.lam_pair, lam_att=args.lam_att,
                               w_mode=args.w_mode, b_far=8.0, w_lead=args.w_lead)
        X0 = s3.initial_states(names=names_sel, offpath=(0.0 if args.onpath else 1.0))
    else:
        names_sel = ("quad_A", "quad_B", "quad_C", "quad_D")[:args.agents]
        cfgs = scn.make_configs(names=names_sel, lam_pair=args.lam_pair, lam_att=args.lam_att,
                                w_mode=args.w_mode, b_far=8.0, w_lead=args.w_lead)
        X0 = scn.initial_states(names=names_sel) if args.onpath else \
            scn.initial_states_offpath(names=names_sel, offset_r=args.offset_r, offset_z=-0.3)
    names = [c.name for c in cfgs]
    paths = {c.name: c.path for c in cfgs}

    inertia = SpatialInertia.MakeFromCentralInertia(
        model.m, np.zeros(3), RotationalInertia(model.Ix, model.Iy, model.Iz))
    builder = DiagramBuilder()
    plant, sg = AddMultibodyPlantSceneGraph(builder, time_step=0.0)
    plant.mutable_gravity_field().set_gravity_vector([0, 0, -model.g])
    sg.AddRenderer("vtk", MakeRenderEngineVtk(RenderEngineVtkParams()))
    sine = args.scenario == "sine"
    # Keep the rendered drone SMALLER than d_s so a d_s-separation does not look like a collision.
    # Skydio mesh: scale 0.0027 -> ~0.43 m; scale 0.0019 -> ~0.30 m (< d_s=0.5 m, clear gap at closest approach).
    qscale = args.qscale
    pcurve = dict(qrange=(0, max(16, 2 * np.pi / getattr(ssc, "W_SINE", 0.5))), n=320, r=args.pr) if sine else dict()
    indices = []
    for c in cfgs:
        b = plant.AddRigidBody(c.name, inertia)
        plant.AddJoint(QuaternionFloatingJoint(c.name + "_j", plant.world_frame(), b.body_frame()))
        add_quad_visual(plant, b, RGB[c.name], scale=qscale)
        add_path_curve(plant, c.name, paths[c.name], RGB[c.name], **pcurve)
        indices.append(b.index())
    plant.Finalize()

    for k, _c in enumerate(cfgs):
        _c.P = args.P; _c.v_max = args.vmax; _c.lam_v = (args.lam_v,) * 3
        if args.vdes: _c.v_des = float(args.vdes[k % len(args.vdes)])
    sysv = builder.AddSystem(RC.V2System(cfgs, indices, model, dt, mode=args.controller))
    builder.Connect(plant.get_state_output_port(), sysv.get_input_port(0))
    builder.Connect(sysv.get_output_port(0), plant.get_applied_spatial_force_input_port())

    # offscreen VTK camera looking at the scene centre
    intr = CameraInfo(args.w, args.h, np.pi / 4)
    cam = ColorRenderCamera(RenderCameraCore("vtk", intr, ClippingRange(0.05, 80.0), RigidTransform()), False)
    if sine:
        _xc = float(np.pi / getattr(ssc, "W_SINE", 0.5))       # crossing is where sin(w x)=0
        tgt = np.array([_xc if args.tgt_x is None else args.tgt_x, 0.0, 0.9])
        eye = tgt + np.array([0.0, -7.5, 5.1])                 # look at the crossing, whatever w is
    elif args.scenario == "three":
        eye = np.array([0.0, -4.5, 4.5]); tgt = np.array([0.0, 0.0, 1.0])   # centred on the origin intersection
    else:
        eye = np.array([4.5, -4.2, 5.4]); tgt = np.array([0.5, 0.5, 1.0])
        if args.tgt_x is not None:
            tgt[0] = args.tgt_x
    # named views orbit the same target at the same distance, so every angle frames the same scene.
    # elevation is capped below 90 deg: looking straight down leaves the camera roll undefined.
    VIEW = {"top": (82.0, -90.0), "side": (18.0, -90.0), "front": (16.0, 0.0)}
    if args.view != "iso" or args.elev is not None or args.azim is not None:
        el, az = VIEW.get(args.view, (35.0, -90.0))
        el = np.radians(args.elev if args.elev is not None else el)
        az = np.radians(args.azim if args.azim is not None else az)
        R = float(np.linalg.norm(eye - tgt))
        eye = tgt + R * np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    if args.tgt_z is not None:
        d = eye - tgt
        tgt = np.array([tgt[0], tgt[1], args.tgt_z])
        eye = tgt + d
    eye = tgt + (eye - tgt) * args.zoom          # closer camera -> larger subjects
    fwd = tgt - eye; fwd /= np.linalg.norm(fwd)
    right = np.cross(fwd, [0, 0, 1])
    if np.linalg.norm(right) < 1e-6:
        right = np.array([1.0, 0.0, 0.0])
    right /= np.linalg.norm(right)
    down = np.cross(fwd, right)
    Rcam = RotationMatrix(np.column_stack([right, down, fwd]))
    Rcw = Rcam.matrix()
    fx, fy, ux, uy = intr.focal_x(), intr.focal_y(), intr.center_x(), intr.center_y()

    def project(pw):
        pc = Rcw.T @ (np.asarray(pw) - eye)
        if pc[2] <= 0.05:
            return None
        return (fx * pc[0] / pc[2] + ux, fy * pc[1] / pc[2] + uy)

    depth_cam = DepthRenderCamera(cam.core(), DepthRange(0.1, 50.0))
    world_fid = plant.GetBodyFrameIdOrThrow(plant.world_body().index())
    sensor = builder.AddSystem(RgbdSensor(world_fid, RigidTransform(Rcam, eye), cam, depth_cam))
    builder.Connect(sg.get_query_output_port(), sensor.query_object_input_port())
    diagram = builder.Build()

    sim = Simulator(diagram); ctx = sim.get_mutable_context()
    RC.set_initial(plant, cfgs, X0, ctx)
    sim.Initialize()
    print(f"Drake VTK render: N={len(cfgs)} circles, {args.w}x{args.h}, rate={args.rate}Hz, "
          f"m={model.m}, Ix={model.Ix}")

    sctx = sensor.GetMyContextFromRoot(ctx)
    plant_ctx = plant.GetMyContextFromRoot(ctx)
    color_port = sensor.color_image_output_port()
    rgb255 = {n: tuple(int(255 * c) for c in RGB[n]) for n in names}
    BG = (204, 229, 255)
    TRAIL = 18
    hist = {n: [] for n in names}; frames = []
    step = 0; t = 0.0
    while t < args.tmax:
        t += dt; step += 1
        sim.AdvanceTo(t)
        if sysv.infeasible:
            print(f"*** INFEASIBLE at t={t:.3f} ***"); break
        if step % args.capture_every == 0:
            xfull = plant.get_state_output_port().Eval(plant_ctx)
            for i, n in enumerate(names):
                s12, _ = RC.parse_agent(xfull, i, len(names))
                hist[n].append(s12[6:9].copy())
            img = color_port.Eval(sctx).data
            im = Image.fromarray(img[:, :, :3].copy())
            d = ImageDraw.Draw(im)
            for n in names:
                proj = [project(p) for p in hist[n][-TRAIL:]]
                for k in range(1, len(proj)):
                    if proj[k - 1] is None or proj[k] is None:
                        continue
                    fade = 0.72 * (1.0 - k / len(proj))
                    col = tuple(int(rgb255[n][c] * (1 - fade) + BG[c] * fade) for c in range(3))
                    d.line([proj[k - 1], proj[k]], fill=col, width=1 + int(2.5 * k / len(proj)))
            d.text((10, 8), f"Drake (VTK)  |  {args.controller}  |  t = {t:5.1f} s  |  N={len(cfgs)}",
                   fill=(20, 20, 20))
            if args.crop_v < 1.0:
                keep = int(round(args.crop_v * im.height))
                top = (im.height - keep) // 2
                im = im.crop((0, top, im.width, top + keep))
            frames.append(im)
    print(f"captured {len(frames)} frames")

    pal = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    qframes = [im.convert("RGB").quantize(palette=pal, dither=Image.NONE) for im in frames]
    out = os.path.join(RESULTS, f"drake_render_{args.scenario}_N{len(cfgs)}_{args.controller}{args.tag}.gif")
    qframes[0].save(out, save_all=True, append_images=qframes[1:], loop=0,
                    duration=int(1000 / args.fps), optimize=True, disposal=2)
    print(f"wrote {out}  ({os.path.getsize(out) / 1024:.0f} KB, {len(qframes)} frames, "
          f"{qframes[0].width}x{qframes[0].height})")


if __name__ == "__main__":
    main()
