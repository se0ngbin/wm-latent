"""Why does recalibration only trade safety for progress? Hypothesis: the plans that matter (safe detours that pass close to the box)
are where h is least separable, and CCEM's all-steps feasibility rule multiplies per-step false alarms. Per start (held-out, safe,
box within 25 steps), P random CEM-style 5-block plans, simulated every env step: unsafe = enters the box; near-safe = never enters,
min joint-space distance to the box < DNEAR (rad); far-safe = otherwise. Veto = CCEM's rule: h - c < 0 on the current frame or any
predicted block (min over 6 positions). Veto rate per group under: uncalibrated, real-frame offset (reacher_calib.py, c on all 6),
per-step predicted offsets (reacher_calib_pred.py), each labeled / safe_q05. Also the in-distribution reference (floor 0, no offset)
and the share of near-safe plans among the safe plans that make progress toward the recorded goal (25 steps ahead).
Env: WMS (wm:ep:headsdir,...), CONDS, NS, P, DNEAR."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E
import reacher_shift as RS

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_jacobian:10:/data/seongbin/lewm/reacher_heads_imag,rc_jac_safeadv_linf:5:/data/seongbin/lewm/reacher_heads_imag_ep5").split(",")]
CONDS = G("CONDS", "floor0,floor0.3,occlusion").split(",")
NS, P, K, FS, DNEAR = int(G("NS", "200")), int(G("P", "64")), 5, 5, float(G("DNEAR", "0.15"))
REAL = json.load(open("/data/seongbin/lewm/reacher_eval_logs/calib/offsets.json"))
PRED = json.load(open("/data/seongbin/lewm/reacher_eval_logs/calib/offsets_pred.json"))
rng = np.random.default_rng(5)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
cs = np.r_[0, np.cumsum(y)]; t = np.arange(len(y) - 26)
ok = (ep[t] >= 9000) & (ep[t] == ep[t + 25]) & (y[t] == 0) & (cs[t + 26] - cs[t] > 0)
st = np.sort(rng.choice(t[ok], NS, replace=False))
Af = A[~np.isnan(A).any(1)]; mu, sd = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((NS, P, 1, 2 * FS)) + 0.7 * rng.standard_normal((NS, P, K, 2 * FS))).astype(np.float32)
def box_dist(q):   # joint-space distance to the box (0 inside); q1 wrapped to (-pi, pi], box q1 in [0, pi], q2 in [-2.88, -2.45]
    q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi
    d1 = 0.0 if 0 <= q1 <= np.pi else min(abs(q1), abs(q1 - np.pi))
    d2 = max(-2.88 - q[1], 0.0, q[1] + 2.45)
    return float(np.hypot(d1, d2))
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env.reset(seed=0)
raw = np.clip(plan.reshape(NS, P, K * FS, 2) * sd + mu, -1, 1)
mind, prog = np.zeros((NS, P)), np.zeros((NS, P))
wrap = lambda d: np.abs((d + np.pi) % (2 * np.pi) - np.pi)
for s, i in enumerate(st):
    g = Q[i + 25]; d0 = wrap(Q[i] - g).max()
    for p in range(P):
        env.set_state(Q[i], V[i]); md = 9.0
        for k in range(K * FS):
            env.step(raw[s, p, k]); md = min(md, box_dist(np.asarray(env.env.physics.data.qpos[:2])))
        mind[s, p] = md; prog[s, p] = 1 - wrap(np.asarray(env.env.physics.data.qpos[:2]) - g).max() / max(d0, 1e-6)
grp = np.where(mind == 0, 0, np.where(mind < DNEAR, 1, 2))   # 0 unsafe, 1 near-safe, 2 far-safe
prg = (grp > 0) & (prog > 0.3)
print(f"{NS} starts x {P} plans: unsafe {np.mean(grp == 0):.3f} near-safe {np.mean(grp == 1):.3f} far-safe {np.mean(grp == 2):.3f} | "
      f"safe plans with progress > 0.3: {prg.mean():.3f}, of which near-safe {np.mean(grp[prg] == 1):.3f}", flush=True)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
def render(cond):
    e = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    fl = cond.startswith("floor")
    e.reset(seed=0, options={"variation_values": {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}} if fl else None)
    sh = "" if fl else cond; RS.apply_physics(e, sh)
    return np.stack([RS.corrupt((e.set_state(Q[i], V[i]), e.render())[1], sh, k) for k, i in enumerate(st)])
imgs = {c: render(c) for c in CONDS}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
for wm, epn, hd in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    heads = {}
    for kind in ["mlp", "linear"]:
        sdict = torch.load(f"{hd}/{wm}/{kind}_s0.pt", map_location=E.DEV)
        if kind == "linear": h = torch.nn.Linear(192, 1); h.load_state_dict(sdict["linear"])
        else:
            h = E.MarginHead(192)
            h.load_state_dict({k_.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k_, v in sdict["margin_gp"].items()})
        heads[kind] = h.to(E.DEV).eval()
    for c in CONDS:
        with torch.no_grad():
            x = (torch.as_tensor(imgs[c], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
            z0 = torch.cat([m.projector(m.encoder(x[i:i + 100], interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, NS, 100)])
            emb = z0.repeat_interleave(P, 0).unsqueeze(1); Ab = torch.from_numpy(plan.reshape(NS * P, K, -1)).to(E.DEV)
            for k in range(K):
                act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        for kind, h in heads.items():
            with torch.no_grad(): hv = h(emb.flatten(0, 1)).view(NS, P, K + 1).cpu().numpy()   # current + 5 predicted
            offs = {"uncal": np.zeros(K + 1)}
            if c != "floor0" and G("OFFSETS", "1") == "1":
                for var in ["labeled", "safe_q05"]:
                    offs[f"real_{var}"] = np.full(K + 1, REAL[f"{wm}_ep{epn}"][kind][c][var])
                    offs[f"pred_{var}"] = np.array(PRED[f"{wm}_ep{epn}"][kind][c][var])
            # plan-level calibration (no offsets per step): drop the current frame (shared by all candidates), threshold the plan score
            # min_k h(zhat_k) at the 5% quantile over SAFE plans of the calibration half (even starts); report on the odd starts
            sc = hv[:, :, 1:].min(-1); cal = np.arange(NS) % 2 == 0
            if c != "floor0":
                cpl = np.quantile(sc[cal][grp[cal] > 0], 0.05); te_ = ~cal
                v = sc[te_] - cpl < 0; gt, pt = grp[te_], prg[te_]
                plan_row = f"plan_q05 (c={cpl:+.2f}, test half): unsafe {v[gt == 0].mean():.2f} near {v[gt == 1].mean():.2f} far {v[gt == 2].mean():.2f} prog-safe {v[pt].mean():.2f}"
            row = [plan_row] if c != "floor0" else []
            for nm, o in offs.items():
                veto = (hv - o).min(-1) < 0
                row.append(f"{nm}: unsafe {veto[grp == 0].mean():.2f} near {veto[grp == 1].mean():.2f} far {veto[grp == 2].mean():.2f} prog-safe {veto[prg].mean():.2f}")
            print(f"VETO {wm} ep{epn} {kind} {c} | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
