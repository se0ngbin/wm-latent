"""Can perturbations too big for the margin function be detected? Label-free detectors per WM and shift, on N held-out states
(episodes >= 9000, half unsafe), each with the frame one action block (5 env steps, dataset actions) later, both rendered under the shift:
  support   kNN distance (k=10, mean) from z to NREF in-distribution training latents (dataset frames, episodes < 9000)
  surprise  ||pred(z_t, a_block) - z_{t+5}||, 1-frame history: the world model's own one-block prediction error
  either    flagged by support or surprise
Thresholds = the 99th percentile on clean (default-appearance) frames of a calibration half of the states; rates are on the other half
(clean false alarm about 0.01 per detector). Per shift: flag rate, AUROC shifted vs clean (paired states), the margin function's bal acc
on the shifted frames (real+imag MLP, $HEADS), and catch = the fraction of its false-safe unsafe states (h >= 0 on an unsafe frame) that
are flagged (does the detector fire on the failures that matter?).
Env: WMS (name:wm:ep,...), N, NREF, FLOORS, SHIFTS (reacher_shift.py names), HEADS. Prints DETECT lines."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm, reacher_shift as RS
from safe_solver import _load_margin_head

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "sigreg:rc_baseline:10,official:lewm_reacher:50,jac:rc_jacobian:10,jac+adv_h:rc_jac_safeadv_linf:10,"
                                      "adv_h varying radius:rc_jac_safeadv_linf_epsmix:10,whole latent:rc_jac_safeadv_latent:10").split(",")]
N, NREF, K, FS, DEV = int(G("N", "1000")), int(G("NREF", "20000")), 10, 5, "cuda"
FLOORS = [a for a in G("FLOORS", "0.1,0.2,0.3,0.4,0.5,0.75,1").split(",") if a]
SHIFTS = G("SHIFTS", "light,light_dir,exposure,white_balance,noise,blur,jpeg,occlusion,cam_zoom,cam_shift,cam_tilt").split(",")   # light = intensity 0.3 (reacher_ood)
HEADS = G("HEADS", "/data/seongbin/lewm/reacher_heads_imag")
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
    ref_i = np.sort(rng.choice(np.where(ep < 9000)[0], NREF, replace=False))
    REF = np.concatenate([f["pixels"][ref_i[i:i + 2000]] for i in range(0, NREF, 2000)])
t = np.arange(len(y) - FS)
ok = (ep[t] >= 9000) & (ep[t] == ep[t + FS]) & ~np.stack([np.isnan(A[t + j]).any(1) for j in range(FS)], 1).any(1)
idx = np.sort(np.r_[rng.choice(t[ok & (y[t] == 1)], N // 2, replace=False), rng.choice(t[ok & (y[t] == 0)], N // 2, replace=False)])
ys = y[idx]
cal = np.zeros(N, bool); cal[rng.choice(N, N // 2, replace=False)] = True; ev = ~cal
Af = A[~np.isnan(A).any(1)]; A_MU, A_SD = Af.mean(0), Af.std(0)
ACT = ((np.stack([A[idx + j] for j in range(FS)], 1) - A_MU) / A_SD).reshape(N, 1, 2 * FS).astype(np.float32)   # one z-scored block
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])

def render(cond):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    var = {"floor.color": (1 - float(cond[5:])) * F0 + float(cond[5:]) * F1} if cond.startswith("floor") else {"light.intensity": np.array([0.3])} if cond == "light" else {}
    env.reset(seed=0, options={"variation_values": var} if var else None)
    sh = cond if cond in RS.APPEARANCE else ""; RS.apply_physics(env, sh)
    out = []
    for d in (0, FS):
        out.append(np.stack([RS.corrupt((env.set_state(Q[i + d], V[i + d]), env.render())[1], sh, k) for k, i in enumerate(idx)]))
    return out
CONDS = ["clean"] + [f"floor{a}" for a in FLOORS] + SHIFTS
IM = {c: render(c) for c in CONDS}
# sanity: the dataset's next frame is the state after the block's actions (so the predictor is asked a real transition)
print(f"N={N} (eval {ev.sum()}) NREF={NREF} | clean render vs dataset frame |diff| "
      f"{np.abs(IM['clean'][0][:20].astype(int) - h5py.File('/data/seongbin/lewm/datasets/reacher.h5', 'r')['pixels'][idx[:20]]).mean():.2f}", flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
def bal(h, lab): sp = h >= 0; return (sp[lab == 0].mean() + (~sp[lab == 1]).mean()) / 2
for name, wm, epn in WMS:
    m = swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(DEV).eval()
    head = _load_margin_head(f"{HEADS}/{wm}/mlp_s0.pt", DEV).eval()
    @torch.no_grad()
    def enc(a, bs=250):
        return torch.cat([m.projector(m.encoder((torch.as_tensor(a[i:i + bs], device=DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD,
                                                 interpolate_pos_encoding=True).last_hidden_state[:, 0]) for i in range(0, len(a), bs)])
    @torch.no_grad()
    def knn(Z, bs=1000): return torch.cat([torch.cdist(Z[i:i + bs], R).topk(K, largest=False).values.mean(1) for i in range(0, len(Z), bs)]).cpu().numpy()
    @torch.no_grad()
    def pred(z): return m.predict(z.unsqueeze(1), m.action_encoder(torch.from_numpy(ACT).to(DEV)))[:, -1]
    R = enc(REF)
    S = {}
    for c in CONDS:
        z0, z5 = enc(IM[c][0]), enc(IM[c][1])
        with torch.no_grad(): h = head(z0).view(-1).cpu().numpy()
        S[c] = dict(sup=knn(z0), sur=(pred(z0) - z5).norm(dim=1).cpu().numpy(), h=h)
    thr = {k: np.quantile(S["clean"][k][cal], 0.99) for k in ("sup", "sur")}
    for c in CONDS:
        s = S[c]; fl = {k: s[k] > thr[k] for k in ("sup", "sur")}; fl["any"] = fl["sup"] | fl["sur"]
        au = {k: roc_auc_score(np.r_[np.zeros(ev.sum()), np.ones(ev.sum())], np.r_[S["clean"][k][ev], s[k][ev]]) for k in ("sup", "sur")}
        fs = ev & (ys == 1) & (s["h"] >= 0)   # false-safe failures on the eval half
        catch = " ".join(f"{k}={fl[k][fs].mean():.3f}" if fs.any() else f"{k}=nan" for k in ("sup", "sur", "any"))
        print(f"DETECT {name} | {c} | bal={bal(s['h'][ev], ys[ev]):.3f} n_falsesafe={fs.sum()} | flag sup={fl['sup'][ev].mean():.3f} "
              f"sur={fl['sur'][ev].mean():.3f} any={fl['any'][ev].mean():.3f} | auroc sup={au['sup']:.3f} sur={au['sur']:.3f} | "
              f"catch {catch} | med ratio sup={np.median(s['sup'][ev]) / np.median(S['clean']['sup'][ev]):.2f} "
              f"sur={np.median(s['sur'][ev]) / np.median(S['clean']['sur'][ev]):.2f}", flush=True)
    del m, R; torch.cuda.empty_cache()
