"""Controller-algebra verification gate: slack direction, row authorities, and the
reduced-QP coefficients over thousands of random admissible states.
Signs are in this repo's z-up convention; the tex is z-down, so p_1, abar_f, abar_1, abar_2 flip."""
import sys, os, numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "tflqp"))
import dynamics, tfl, optimizer
from scipy.spatial.transform import Rotation

def run(n=4000, seed=0, verbose=True):
    rng = np.random.default_rng(seed)
    mdl = dynamics.Model(); m, Ix, Iy, Iz = mdl.m, mdl.Ix, mdl.Iy, mdl.Iz
    E = tfl.E_SEL
    keys = ["Bq_col4","Bq_p","p1","p_tau1","p_tau2","p_tau3","abar_f","abar_1","abar_2",
            "abar_v","abar_ij","slack_dir","dhat"]
    worst = {k: 0.0 for k in keys}; ok_n = 0
    for _ in range(n):
        x = np.zeros(14)
        x[0:3] = rng.uniform(-1.0, 1.0, 3); x[3:6] = rng.normal(0, 1.5, 3)
        x[6:9] = np.array([1.5, 0., 1.]) + rng.normal(0, .4, 3); x[9:12] = rng.normal(0, .5, 3)
        x[12] = m * 9.8 + rng.normal(0, 2.0); x[13] = rng.normal(0, 1.0)
        if abs(np.cos(x[0])) < .2 or abs(np.cos(x[1])) < .2 or x[12] <= 1e-3: continue
        Bq = mdl.Bq(x); Jg = rng.normal(0, 1, (3, 3))
        if abs(np.linalg.det(Jg)) < 1e-2: continue
        D = np.vstack([Jg @ Bq, mdl.yaw_row(x)])
        if np.linalg.cond(D) > 1e11: continue
        p = np.linalg.solve(D, E); w = np.linalg.solve(Jg, np.array([0., 0., 1.]))
        R = Rotation.from_euler("ZYX", [x[2], x[1], x[0]]).as_matrix()
        u = lambda k, v: worst.__setitem__(k, max(worst[k], v))
        u("Bq_col4", np.max(np.abs(Bq[:, 3])) / max(1.0, np.max(np.abs(Bq))))
        u("Bq_p",    np.max(np.abs(Bq @ p - w)))
        u("p1",      abs(p[0] - m * np.dot(R[:, 2], w)))
        u("p_tau1",  abs(p[1] + (m * Ix / x[12]) * np.dot(R[:, 1], w)))
        u("p_tau2",  abs(p[2] - (m * Iy / x[12]) * np.dot(R[:, 0], w)))
        u("p_tau3",  abs(p[3] + (Iz / Iy) * np.tan(x[0]) * p[2]))
        u("abar_f",  abs(-np.array([1., 0, 0, 0]) @ p + p[0]))
        LgLf_phi = np.array([0, 1/Ix, np.sin(x[0])*np.tan(x[1])/Iy, np.cos(x[0])*np.tan(x[1])/Iz])
        LgLf_th  = np.array([0, 0, np.cos(x[0])/Iy, -np.sin(x[0])/Iz])
        for st in (1., -1.):
            u("abar_1", abs(-(-st*LgLf_phi) @ p - st*p[1]/Ix))
            u("abar_2", abs(-(-st*LgLf_th) @ p - st*p[2]/(Iy*np.cos(x[0]))))
            u("abar_v", abs(-(-st*D[2, :]) @ p - st))
        xij = rng.normal(0, 1, 3)
        u("abar_ij", abs(-(2*xij @ Bq) @ p + 2*xij @ w))
        v = rng.normal(0, 3, 4); nu0 = np.linalg.solve(D, v); d = rng.normal()
        u("slack_dir", np.max(np.abs(D @ (nu0 + p*d) - (v + E*d))))
        W = np.eye(4); P = 1.0
        r = optimizer.reduce_and_solve(np.zeros((0, 4)), np.zeros(0), nu0, p, W, P)
        u("dhat", abs(r["delta_hat"] + (p @ W @ nu0)/(p @ W @ p + P)))
        ok_n += 1
    tol = {"Bq_col4":1e-15,"Bq_p":1e-8,"p1":1e-8,"p_tau1":1e-9,"p_tau2":1e-9,"p_tau3":1e-9,
           "abar_f":1e-9,"abar_1":1e-8,"abar_2":1e-8,"abar_v":1e-8,"abar_ij":1e-8,
           "slack_dir":1e-8,"dhat":1e-12}
    lbl = {"Bq_col4":"B_q tau_3 column is zero (relative)","Bq_p":"B_q p = J_g^-1 e3   [Lemma 6, frame-inv]",
           "p1":"p_1    = +m<R e3,w>   [z-up; tex -]","p_tau1":"p_tau1 = -(mIx/x13)<R e2,w> [z-up; tex +]",
           "p_tau2":"p_tau2 = +(mIy/x13)<R e1,w> [z-up; tex -]","p_tau3":"p_tau3 = -(Iz/Iy)tan(phi)p_tau2 [inv]",
           "abar_f":"abar_f = -p_1","abar_1":"abar_1^{i,s} = s p_tau1/Ix",
           "abar_2":"abar_2^{i,s} = s p_tau2/(Iy cos phi)","abar_v":"abar_v^{i,s} = s exactly   [frame-inv]",
           "abar_ij":"abar_ij = -2<x_ij,w>       [frame-inv]","slack_dir":"slack perturbs ONLY row 3 of D nu = v",
           "dhat":"dhat = -(p^T W nu_0)/Q"}
    allok = True
    if verbose: print(f"verified over {ok_n} random admissible states\n")
    for k in keys:
        good = worst[k] < tol[k]; allok &= good
        if verbose: print(f"  [{'PASS' if good else 'FAIL'}] {lbl[k]:42s} {worst[k]:.2e} < {tol[k]:.0e}")
    if verbose: print("\n" + ("ALL CHECKS PASS" if allok else "*** FAILURES ***"))
    return allok

if __name__ == "__main__":
    sys.exit(0 if run() else 1)
