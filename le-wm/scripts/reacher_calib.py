"""Test-time threshold recalibration of a Reacher margin function under a shift (no retraining). For each condition (floor<a> or a
reacher_shift.py appearance shift), calibration frames = held-out states (episodes >= 9000) DISJOINT from the test states, rendered
under the condition; joint-box labels from qpos (a robot gets them from its joint encoders). Offsets c (use h - c):
  labeled   : 100 safe + 100 unsafe calibration frames; c = the threshold maximizing balanced accuracy on them
  safe_only : 100 safe calibration frames only (no failure data); c = median_h(shifted safe) - median_h(base safe), i.e. undo the
              shift of h on safe states
  safe_q05  : 100 safe calibration frames only; c = 5% quantile of h on them (anything below the bulk of safe states reads unsafe)
Static check on the 5000 test states (reacher_ood.py's set): bal acc / AUC before and after. Writes offsets to OUT (json,
{wm: {head: {cond: {labeled: c, safe_only: c}}}}) for reacher_cem.py (H_OFFSET).
Env: WMS (wm:ep:headsdir,...), CONDS, NCAL, OUT."""
import os, sys, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E
import reacher_shift as RS

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10:/data/seongbin/lewm/reacher_heads_imag,rc_jacobian:10:/data/seongbin/lewm/reacher_heads_imag,"
                                      "rc_jac_safeadv_linf:5:/data/seongbin/lewm/reacher_heads_imag_ep5").split(",")]
CONDS = G("CONDS", "floor0.3,floor0.4,floor0.5,exposure,white_balance,cam_shift,cam_tilt,occlusion").split(",")
NCAL, OUT = int(G("NCAL", "100")), G("OUT", "/data/seongbin/lewm/reacher_eval_logs/calib/offsets.json")
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    ep, Q, V, y = f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["failures"][:].astype(int)
ho = np.where(ep >= 9000)[0]
te = np.sort(np.random.default_rng(0).choice(ho, 5000, replace=False))            # = reacher_ood.py test states
rest = np.setdiff1d(ho, te); r = np.random.default_rng(1)
cs, cu = r.choice(rest[y[rest] == 0], NCAL, replace=False), r.choice(rest[y[rest] == 1], NCAL, replace=False)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
def render(cond, idx):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    fl = cond.startswith("floor") and cond != "floor0"
    env.reset(seed=0, options={"variation_values": {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1}} if fl else None)
    sh = "" if cond.startswith("floor") else cond; RS.apply_physics(env, sh)
    return np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(idx)])
imgs = {c: {"test": render(c, te), "cs": render(c, cs), "cu": render(c, cu)} for c in ["floor0"] + CONDS}
print(f"calibration {NCAL} safe + {NCAL} unsafe, test 5000 (unsafe {y[te].mean():.3f})", flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
def bal(h, lab): sp = h >= 0; return (sp[lab == 0].mean() + (~sp[lab == 1]).mean()) / 2
out = json.load(open(OUT)) if os.path.exists(OUT) else {}
for wm, epn, hd in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    @torch.no_grad()
    def enc(a, bs=250):
        return torch.cat([m.projector(m.encoder(((torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD),
                                                 interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), bs)])
    Z = {c: {k: enc(v) for k, v in d.items()} for c, d in imgs.items()}
    for kind in ["mlp", "linear"]:
        sd = torch.load(f"{hd}/{wm}/{kind}_s0.pt", map_location=E.DEV)
        if kind == "linear": h = torch.nn.Linear(192, 1); h.load_state_dict(sd["linear"])
        else:
            h = E.MarginHead(192)
            h.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): v for k, v in sd["margin_gp"].items()})
        h = h.to(E.DEV).eval(); H = lambda z: h(z).detach().cpu().numpy().ravel()
        base_safe = np.median(H(Z["floor0"]["cs"]))
        for c in CONDS:
            ht, hs_, hu_ = H(Z[c]["test"]), H(Z[c]["cs"]), H(Z[c]["cu"])
            cand = np.r_[hs_, hu_]; lab = np.r_[np.zeros(NCAL), np.ones(NCAL)].astype(int)
            c_lab = float(max(np.unique(cand), key=lambda t: bal(cand - t, lab)))
            c_safe = float(np.median(hs_) - base_safe)
            c_q = float(np.quantile(hs_, 0.05))   # safe-only: threshold at the 5% quantile of h on shifted SAFE frames
            yt = y[te]
            print(f"CALIB {wm} ep{epn} {kind} {c}: auc={roc_auc_score(1 - yt, ht):.3f} bal raw={bal(ht, yt):.3f} "
                  f"labeled={bal(ht - c_lab, yt):.3f} (c={c_lab:+.3f}) safe_only={bal(ht - c_safe, yt):.3f} (c={c_safe:+.3f}) "
                  f"safe_q05={bal(ht - c_q, yt):.3f} (c={c_q:+.3f})", flush=True)
            out.setdefault(f"{wm}_ep{epn}", {}).setdefault(kind, {})[c] = {"labeled": c_lab, "safe_only": c_safe, "safe_q05": c_q}
    del m; torch.cuda.empty_cache()
os.makedirs(os.path.dirname(OUT), exist_ok=True); json.dump(out, open(OUT, "w"), indent=1)
