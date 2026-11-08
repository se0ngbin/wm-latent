"""Per-step threshold recalibration on PREDICTED latents (CCEM reads h on the current frame + K predicted blocks; real-frame
offsets from reacher_calib.py over-tighten the predicted, partly re-grounded, states). Per condition (floor<a> or a reacher_shift
appearance shift) and WM: calibration starts = held-out frames (episodes >= 9000 not used by the CEM pairs) rendered under the
condition, PPS random CEM-style 5-block plans per start (as reacher_margin_imag), predicted latents zhat_k (1-frame history),
labels = the plan replayed in the env (joint box at block end; a robot gets them by executing plans and reading its encoders).
Offsets c_k, k = 1..K (use h(zhat_k) - c_k):
  labeled  : per k, the threshold with the best balanced accuracy over safe + unsafe predicted states
  safe_q05 : per k, the 5% quantile of h over predicted states whose executed state was SAFE (no failure data)
c_0 (current real frame) is taken from reacher_calib.py's offsets (same variant). Static check: per-step bal acc on held-out
calibration-disjoint plans, uncalibrated vs calibrated. Writes {wm_ep: {head: {cond: {variant: [c_0..c_K]}}}} to OUT.
Env: WMS (wm:ep:headsdir,...), CONDS, NCAL (starts), PPS, OUT, REAL (reacher_calib offsets)."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E
import reacher_shift as RS

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_jacobian:10:/data/seongbin/lewm/reacher_heads_imag,rc_jac_safeadv_linf:5:/data/seongbin/lewm/reacher_heads_imag_ep5").split(",")]
CONDS = G("CONDS", "floor0.3,floor0.4,occlusion,cam_shift").split(",")
NCAL, PPS, K, FS, GOFF = int(G("NCAL", "300")), int(G("PPS", "4")), 5, 5, 25
OUT = G("OUT", "/data/seongbin/lewm/reacher_eval_logs/calib/offsets_pred.json")
REAL = json.load(open(G("REAL", "/data/seongbin/lewm/reacher_eval_logs/calib/offsets.json")))
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    fl, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
cs = np.r_[0, np.cumsum(fl)]; t = np.arange(len(fl) - GOFF)                     # reacher_cem.py data-mode pairs -> exclude their episodes
ok = (ep[t] >= 9000) & (ep[t] == ep[t + GOFF]) & (fl[t] == 0) & (fl[t + GOFF] == 0) & (cs[t + GOFF + 1] - cs[t] > 0)
cand = t[ok]; sel = cand[np.random.default_rng(42).choice(len(cand) - 1, size=50, replace=False)]
pool = t[(ep[t] >= 9000) & ~np.isin(ep[t], ep[sel]) & (ep[t] == ep[t + GOFF])]
near = pool[cs[pool + GOFF + 1] - cs[pool] > 0]                                  # half the starts near the box
rng = np.random.default_rng(3)
st = np.sort(np.r_[rng.choice(near, NCAL // 2, replace=False), rng.choice(pool, NCAL - NCAL // 2, replace=False)])
Af = A[~np.isnan(A).any(1)]; mu, sd = Af.mean(0), Af.std(0)
plan = (rng.standard_normal((NCAL, PPS, 1, 2 * FS)) + 0.7 * rng.standard_normal((NCAL, PPS, K, 2 * FS))).astype(np.float32)
env0 = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped; env0.reset(seed=0)
def in_box(q): q1 = (q[0] + np.pi) % (2 * np.pi) - np.pi; return 0 <= q1 <= np.pi and -2.88 <= q[1] <= -2.45
lab = np.zeros((NCAL, PPS, K), int); raw = np.clip(plan.reshape(NCAL, PPS, K * FS, 2) * sd + mu, -1, 1)
for s, i in enumerate(st):
    for p in range(PPS):
        env0.set_state(Q[i], V[i])
        for k in range(K * FS):
            env0.step(raw[s, p, k])
            if k % FS == FS - 1: lab[s, p, k // FS] = in_box(np.asarray(env0.env.physics.data.qpos[:2]))
half = np.arange(NCAL) % 2 == 0                                                   # even starts calibrate, odd starts test
print(f"{NCAL} starts x {PPS} plans; unsafe predicted-state frac per step {np.round(lab.mean((0, 1)), 3).tolist()}", flush=True)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
def render(cond):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    fl_ = cond.startswith("floor")
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}} if fl_ else None)
    sh = "" if fl_ else cond; RS.apply_physics(env, sh)
    return np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(st)])
imgs = {c: render(c) for c in CONDS}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
def bal(h, y): sp = h >= 0; return (sp[y == 0].mean() + (~sp[y == 1]).mean()) / 2
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
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
            z0 = torch.cat([m.projector(m.encoder(x[i:i + 100], interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, NCAL, 100)])
            emb = z0.repeat_interleave(PPS, 0).unsqueeze(1); Ab = torch.from_numpy(plan.reshape(NCAL * PPS, K, -1)).to(E.DEV)
            for k in range(K):
                act = m.action_encoder(Ab[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        for kind, h in heads.items():
            with torch.no_grad(): hv = h(emb[:, 1:].flatten(0, 1)).view(NCAL, PPS, K).cpu().numpy()
            res = {}
            for var in ["labeled", "safe_q05"]:
                cks, before, after = [], [], []
                for k in range(K):
                    hc, yc = hv[half, :, k].ravel(), lab[half, :, k].ravel(); ht, yt = hv[~half, :, k].ravel(), lab[~half, :, k].ravel()
                    ck = float(max(np.unique(hc), key=lambda th: bal(hc - th, yc))) if var == "labeled" else float(np.quantile(hc[yc == 0], 0.05))
                    cks.append(ck); before.append(bal(ht, yt)); after.append(bal(ht - ck, yt))
                res[var] = [REAL[f"{wm}_ep{epn}"][kind][c][var]] + cks
                print(f"CALIBPRED {wm} ep{epn} {kind} {c} {var}: c_1..5={' '.join(f'{v:+.2f}' for v in cks)} (real-frame c_0 {res[var][0]:+.2f}) | "
                      f"held-out bal per step raw {' '.join(f'{v:.3f}' for v in before)} -> {' '.join(f'{v:.3f}' for v in after)}", flush=True)
            out.setdefault(f"{wm}_ep{epn}", {}).setdefault(kind, {})[c] = res
    del m; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
