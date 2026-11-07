"""Zero-shot appearance OOD for Reacher margin functions (reacher_margin.py heads, trained on default-appearance frames of episodes < 9000).
5000 states of held-out episodes (>= 9000, natural ratio), each rendered in the env under: base (default appearance; checked against the
dataset frame), arm color (default [0.7, 0.5, 0.3] -> purple), floor color (blue checker -> brown), light (intensity 0.7 -> 0.3).
The joint box is invisible, so every view has the same labels. AUC, balanced sign acc ((TPR_safe + TPR_unsafe) / 2), predicted-safe frac.
Env: WMS (comma list), EP, HEADS, N."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

WMS, EP = os.environ.get("WMS", "lewm_reacher").split(","), os.environ.get("EP", "50")
HEADS, N = os.environ.get("HEADS", "/data/seongbin/lewm/reacher_heads"), int(os.environ.get("N", "5000"))
VIEWS = {"base": {}, "arm_color": {"agent.color": np.array([0.55, 0.2, 0.75])},
         "floor_color": {"floor.color": np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])}, "light": {"light.intensity": np.array([0.3])}}
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    ep = f["ep_idx"][:]; idx = np.sort(np.random.default_rng(0).choice(np.where(ep >= 9000)[0], N, replace=False))
    Q, V, y, ref = f["qpos"][:][idx], f["qvel"][:][idx], f["failures"][:][idx].astype(int), f["pixels"][idx[:20]]
env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
imgs = {}
for v, vals in VIEWS.items():
    env.reset(seed=0, options={"variation_values": vals} if vals else None)
    out = []
    for q, qd in zip(Q, V): env.set_state(q, qd); out.append(np.ascontiguousarray(env.render()))
    imgs[v] = np.stack(out)
print(f"N={N} unsafe={y.mean():.3f} | base render vs dataset frame: mean |diff| {np.abs(imgs['base'][:20].astype(int) - ref).mean():.2f} "
      f"| view vs base mean |diff|: " + " ".join(f"{v}={np.abs(imgs[v].astype(int) - imgs['base']).mean():.1f}" for v in VIEWS), flush=True)
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
for wm in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{EP}.pt").to(E.DEV).eval()
    def enc(a, bs=256):
        out = []
        for i in range(0, len(a), bs):
            x = (torch.as_tensor(a[i:i + bs], device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
            with torch.no_grad(): out.append(m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]))
        return torch.cat(out)
    Z = {v: enc(a) for v, a in imgs.items()}
    for hk in ["mlp", "linear"]:
        sd = torch.load(f"{HEADS}/{wm}/{hk}_s0.pt", map_location=E.DEV)
        if hk == "linear": head = torch.nn.Linear(192, 1).to(E.DEV); head.load_state_dict(sd["linear"])
        else:
            head = E.MarginHead(192).to(E.DEV)
            head.load_state_dict({k.replace("layers.Margin GP_linear", "l.l").replace("layers.Margin GP_norm", "l.n").replace("mean_layer.", "m."): t for k, t in sd["margin_gp"].items()})
        head.eval(); row = []
        for v in VIEWS:
            with torch.no_grad(): h = head(Z[v]).cpu().numpy().ravel()
            sp = h >= 0; row.append(f"{v}: auc={roc_auc_score(1 - y, h):.3f} bal={(sp[y == 0].mean() + (~sp[y == 1]).mean()) / 2:.3f} predsafe={sp.mean():.3f}")
        print(f"REACHER_OOD {wm} {hk} | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
