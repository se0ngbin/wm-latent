"""Label-free test-time BatchNorm adaptation for a Reacher LeWM (port of imag-margin le-wm/scripts/bn_adapt_save.py, the Dubins
procedure): projector BN statistics recomputed (reset, cumulative average) from NADAPT unlabeled frames rendered under the
condition, then the predictor output projection's BN statistics from NB-block random CEM-style imagined rollouts (5 z-scored
2*FS-dim blocks, as reacher_margin_imag) from those adapted latents. Nothing else changes; the margin functions stay the ones
trained on <wm>. Adaptation frames: held-out episodes (>= 9000) excluding the reacher_cem.py data-mode pair episodes; SAFEONLY=1
keeps only safe frames (a robot cannot collect failures). Writes checkpoints/<wm>_bn<cond><BNTAG>/weights_epoch_<EP>.pt + config
links, and links reacher_heads_imag/<out> -> reacher_heads_imag/<wm> so every Reacher script loads the original heads.
Conditions: floor<a>, light, arm_color, or any reacher_shift.py appearance shift.
Env: NADAPT (2000), NB (5), SAFEONLY (0/1), BNSETS (proj | pred | both), BNTAG, HEADS. Usage: reacher_bn_adapt.py <wm> <ep> <cond>"""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
import reacher_shift as RS

wm, ep_, cond = sys.argv[1], sys.argv[2], sys.argv[3]
G = os.environ.get
NADAPT, NB, FS, SAFEONLY, DEV = int(G("NADAPT", "2000")), int(G("NB", "5")), 5, G("SAFEONLY", "0") == "1", "cuda"
HEADS = G("HEADS", "/data/seongbin/lewm/reacher_heads_imag")
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    fl, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
cs = np.r_[0, np.cumsum(fl)]; t = np.arange(len(fl) - 25)
ok = (ep[t] >= 9000) & (ep[t] == ep[t + 25]) & (fl[t] == 0) & (fl[t + 25] == 0) & (cs[t + 26] - cs[t] > 0)
cand = t[ok]; sel = cand[np.random.default_rng(42).choice(len(cand) - 1, size=50, replace=False)]   # reacher_cem.py pairs
pool = np.where((ep >= 9000) & ~np.isin(ep, ep[sel]))[0]
if SAFEONLY: pool = pool[fl[pool] == 0]
ra = np.random.default_rng(11)
idx = np.sort(ra.choice(pool, NADAPT, replace=False))
Af = A[~np.isnan(A).any(1)]; mu, sd = Af.mean(0), Af.std(0)
plan = torch.from_numpy((ra.standard_normal((NADAPT, 1, 2 * FS)) + 0.7 * ra.standard_normal((NADAPT, NB, 2 * FS))).astype(np.float32)).to(DEV)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
var = {}
if cond.startswith("floor"): var = {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}
elif cond == "light": var = {"light.intensity": np.array([0.3])}
elif cond == "arm_color": var = {"agent.color": np.array([0.55, 0.2, 0.75])}
env.reset(seed=0, options={"variation_values": var} if var else None)
sh = cond if cond in RS.APPEARANCE else ""; RS.apply_physics(env, sh)
frames = np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(idx)])
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
m = swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{ep_}.pt").to(DEV).eval()
def stage(mod):
    bns = [b for b in mod.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
    for b in bns: b.reset_running_stats(); b.momentum = None; b.train()
    return bns
@torch.no_grad()
def enc(a, bs=250):
    return torch.cat([m.projector(m.encoder((torch.as_tensor(a[i:i + bs], device=DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD,
                                            interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), bs)])
BNSETS = G("BNSETS", "both")   # proj | pred | both: which BatchNorm set to re-estimate (the other keeps its training statistics)
assert BNSETS in ("proj", "pred", "both"), BNSETS
with torch.no_grad():
    if BNSETS in ("proj", "both"):
        bns = stage(m.projector); enc(frames)
        for b in bns: b.eval()
    z = enc(frames).unsqueeze(1)
    bns = stage(m.pred_proj) if BNSETS in ("pred", "both") else []
    for i in range(0, NADAPT, 250):
        e = z[i:i + 250]
        for k in range(NB):
            e = torch.cat([e, m.predict(e[:, -3:], m.action_encoder(plan[i:i + 250, :k + 1])[:, -3:])[:, -1:]], 1)
    for b in bns: b.eval()
root = os.environ["STABLEWM_HOME"] + "/checkpoints"; src = os.path.realpath(f"{root}/{wm}")
name = f"{wm}_bn{cond}" + G("BNTAG", ""); out = f"{root}/{name}"
os.makedirs(out, exist_ok=True); torch.save(m.state_dict(), f"{out}/weights_epoch_{ep_}.pt")
for c in ("config.json", "config.yaml"):
    if os.path.exists(f"{src}/{c}") and not os.path.exists(f"{out}/{c}"): os.symlink(f"{src}/{c}", f"{out}/{c}")
if not os.path.exists(f"{HEADS}/{name}"): os.symlink(f"{HEADS}/{wm}", f"{HEADS}/{name}")
# safe-only threshold for the adapted model (as the Dubins bn_safe_cem.sh): c = Q-quantile of h over the SAFE adaptation frames
import json
from safe_solver import _load_margin_head
Qc = float(G("QTHR", "0.1")); safe = fl[idx] == 0; th = {}
with torch.no_grad():
    za = enc(frames)[torch.as_tensor(safe, device=DEV)]
    for kind in ["mlp", "linear"]:
        sdict = torch.load(f"{HEADS}/{wm}/{kind}_s0.pt", map_location=DEV)
        if kind == "linear": hh = torch.nn.Linear(za.shape[1], 1).to(DEV); hh.load_state_dict(sdict["linear"])
        else: hh = _load_margin_head(f"{HEADS}/{wm}/{kind}_s0.pt", DEV).eval()
        th[kind] = float(torch.quantile(hh(za).float().view(-1), Qc))
J = "/data/seongbin/lewm/reacher_eval_logs/bn/thresholds.json"; os.makedirs(os.path.dirname(J), exist_ok=True)
d = json.load(open(J)) if os.path.exists(J) else {}; d[name] = th; json.dump(d, open(J, "w"), indent=1)
print(f"safe-only q{Qc:.2f} thresholds {th}", flush=True)
print(f"saved {out}/weights_epoch_{ep_}.pt ({NADAPT} {'safe ' if SAFEONLY else ''}frames, cond {cond}; heads -> {HEADS}/{wm})", flush=True)
