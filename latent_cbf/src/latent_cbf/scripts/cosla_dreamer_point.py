"""Dreamer datapoint for the cos_la-vs-isotropy figure. Same protocol as cosla_vs_jac.py (LEWM):
per-model Jacobian spectrum + cos_la(color,rotate) with the failure (inside-obstacle) axis, but for
the dubins Dreamer RSSM. x-axis is sigma1-share (=lambda1/sum lambda, dimension-invariant) so it's
comparable to the ViT encoders despite different feat dim / architecture. Reuses build_dreamer_wm,
render_cond, feats, make_env from critic_ood_eval_dreamer."""
import sys
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import gym as old_gym  # noqa
import gymnasium as gym  # noqa
from configs import DreamerConfig, Config
from compare_critic_to_hj_dreamer import build_dreamer_wm
from critic_ood_eval_dreamer import render_cond, feats, make_env

DEV = "cuda:0"
RSSM = "/data/seongbin/dreamer/dreamer/enc_lip_sweep/baseline/rssm_ckpt.pt"
OBSTACLES = [(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)]
WB = (-1.5, 1.5, -1.5, 1.5)
N, NF = 3000, 4
AXES = {
    "color":  (dict(obstacle="red", shape="circle", rot=0), dict(obstacle="purple", shape="circle", rot=0)),
    "rotate": (dict(obstacle="red", shape="circle", rot=0), dict(obstacle="red", shape="circle", rot=90)),
}

def inside_obstacle(xy):
    hit = np.zeros(len(xy), bool)
    for cx, cy, r in OBSTACLES:
        hit |= (np.hypot(xy[:, 0]-cx, xy[:, 1]-cy) <= r)
    return hit.astype(np.float32)

def cosla(mu, la):
    return float(abs((mu@la)/(np.linalg.norm(mu)*np.linalg.norm(la)+1e-12)))

def jac_svd(wm, img, th):
    """svdvals of d feat / d image for one frame. img: (H,W,3) uint8 array."""
    x = torch.tensor(img[None, None], dtype=torch.float32, device=DEV).requires_grad_(True)  # (1,1,H,W,3)
    obs_state = np.stack([np.cos([th]), np.sin([th])], -1)[:, None].astype(np.float32)
    batch = {"image": img[None, None].astype(np.float32), "obs_state": obs_state,
             "action": np.zeros((1,1,1), np.float32), "is_first": np.ones((1,1,1), np.float32),
             "is_terminal": np.zeros((1,1,1), np.float32)}
    data = wm.preprocess(batch)
    data["image"] = x / 255.0   # splice differentiable image back in (preprocess's torch.tensor() detaches)
    embed = wm.encoder(data)
    states, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
    z = wm.dynamics.get_feat(states)[0, -1]   # (D,)
    J = torch.stack([torch.autograd.grad(z[k], x, retain_graph=True)[0].flatten().detach()
                     for k in range(z.numel())])
    return torch.linalg.svdvals(J.float()).cpu().numpy()

def main():
    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, RSSM, DEV)
    env = make_env()
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(WB[0],WB[1],N), rng.uniform(WB[2],WB[3],N),
                   rng.uniform(-np.pi,np.pi,N)], 1).astype(np.float32)
    # Jacobian spectrum (avg over frames)
    S = np.stack([jac_svd(wm, render_cond(env, st[i:i+1], AXES["color"][0])[0], st[i,2]) for i in range(NF)]).mean(0)
    lam = S**2; effrank = float((lam.sum()**2)/(lam**2).sum()); s1share = float(lam[0]/lam.sum())
    y = inside_obstacle(st[:, :2])
    cl = {}
    for axis, (og, new) in AXES.items():
        Zog = feats(wm, render_cond(env, st, og), st[:, 2]).cpu().numpy()
        Znew = feats(wm, render_cond(env, st, new), st[:, 2]).cpu().numpy()
        mu = (Znew - Zog).mean(0); la = Zog[y==1].mean(0) - Zog[y==0].mean(0)
        cl[axis] = cosla(mu, la)
    print(f"CJ dreamer    orig      effrank={effrank:6.1f} froJ={float(np.sqrt(lam.sum())):7.2f} "
          f"s1share={s1share:.3f} cos_la_color={cl['color']:.3f} cos_la_rot={cl['rotate']:.3f} featdim={len(lam)}")
    print("CJ_DREAMER_DONE")

if __name__ == "__main__":
    main()
