"""Paired appearance-shift latent probe: for FIXED states, how does the frozen
encoder move a red-obstacle image vs its perturbed version, and does that move
matter to the frozen margin? Explains what makes zero-shot work.

Pairing on state isolates pure appearance. In the head's readout space:
  Z0 = z(og appearance),  Z1 = z(new appearance),  D = Z1 - Z0.
Metrics (sigma = RMS pairwise distance in the Z0 cloud, the intrinsic scale):
  displ   = mean||D|| / sigma            raw invariance (smaller = closer)
  transl  = ||mean D|| / sigma           global recolor offset (bias-fixable)
  deform  = RMS||D - mean D|| / sigma     state-dependent warp (NOT bias-fixable)
  corr_h  = corr(h(Z0), h(Z1))           does the margin ranking survive?
  slope,bias of h1 ~ a*h0 + b            pure threshold shift (a~1,b!=0) vs warp
  dbias   = mean(h1-h0) / std(h0)        margin shift in units of its own spread
  flip    = sign-flip rate for |h0|<eps  near-boundary damage (what a CBF feels)
  cosabs  = mean |cos(D, grad_z h)|      is the shift in the dangerous direction?
  cka     = linear CKA(Z0, Z1)           info linearly recoverable at all?
Usage: latent_shift_probe.py --backend lewm --enc <ck> --margin <m> --axis color
       latent_shift_probe.py --backend dreamer --rssm <ck> --axis color
"""
import sys, argparse
from pathlib import Path
import numpy as np, torch
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path: sys.path.append(_p)
import gym as old_gym  # noqa
import gymnasium as gym  # noqa
from sklearn.metrics import roc_auc_score
from configs import DreamerConfig, Config
from critic_ood_eval import render_cond as jepa_render, feats as jepa_feats, CONDS, DEV
from latent_cbf.dubins.dubins_env import DubinsEnv

N = 6000
AXES = {"color": "purple", "shape": "diamond", "rotate": "rot90"}


def inside_obstacle(xy, shape, obstacles):
    hit = np.zeros(len(xy), bool)
    for cx, cy, r in obstacles:
        dx, dy = np.abs(xy[:, 0] - cx), np.abs(xy[:, 1] - cy)
        d = (dx + dy) if shape == "diamond" else np.hypot(dx, dy)
        hit |= (d <= r)
    return hit.astype(np.float32)


def sigma_pairwise(Z):  # RMS pairwise distance = sqrt(2 * total variance)
    return float(np.sqrt(2.0 * Z.var(0).sum()))


def linear_cka(X, Y):
    X = X - X.mean(0); Y = Y - Y.mean(0)
    num = (Y.T @ X) ** 2
    return float(num.sum() / (np.linalg.norm(X.T @ X) * np.linalg.norm(Y.T @ Y) + 1e-12))


def head_and_grad(head, Z):
    """h(Z) and grad_Z h(Z), Z a numpy array -> (h [N], g [N,D])."""
    t = torch.tensor(Z, device=DEV, requires_grad=True)
    out = head(t).view(-1)
    g, = torch.autograd.grad(out.sum(), t)
    return out.detach().cpu().numpy(), g.detach().cpu().numpy()


def retrieval(Z0, Z1, nq=2000):
    """For new-appearance queries, rank of their OWN red state in the red gallery.
    Parameter-free 'does purple land on its own red spot': nn1 = frac rank-0,
    medrank = median rank (0 = perfect), both over a random subset of queries."""
    n = Z0.shape[0]; nq = min(nq, n)
    qi = np.random.default_rng(0).choice(n, nq, replace=False)
    g = torch.tensor(Z0, device=DEV)
    q = torch.tensor(Z1[qi], device=DEV)
    d = torch.cdist(q, g)                                  # nq x n
    true_d = d[torch.arange(nq), torch.tensor(qi, device=DEV)]
    rank = (d < true_d[:, None]).sum(1).cpu().numpy()      # #gallery strictly closer
    return float((rank == 0).mean()), float(np.median(rank)), nq


def metrics(Z0, Z1, h0, g0, h1, y):
    D = Z1 - Z0
    sig = sigma_pairwise(Z0)
    mu = D.mean(0)
    r = D - mu
    dnorm = np.linalg.norm(D, axis=1)
    displ = float(dnorm.mean()) / sig
    transl = float(np.linalg.norm(mu)) / sig
    deform = float(np.sqrt((r ** 2).sum(1).mean())) / sig
    # normalized distances: vs safe/unsafe class gap, and cosine similarity
    safe_m = Z0[y == 0].mean(0); unsafe_m = Z0[y == 1].mean(0)
    classgap = float(np.linalg.norm(unsafe_m - safe_m)) + 1e-12
    displ_cg = float(dnorm.mean()) / classgap
    cossim = float((( (Z0 * Z1).sum(1)) / (np.linalg.norm(Z0, axis=1) * np.linalg.norm(Z1, axis=1) + 1e-12)).mean())
    nn1, medrank, nq = retrieval(Z0, Z1)
    corr = float(np.corrcoef(h0, h1)[0, 1])
    a, b = np.polyfit(h0, h1, 1)
    dbias = float((h1 - h0).mean() / (h0.std() + 1e-9))
    eps = float(np.quantile(np.abs(h0), 0.25))  # near-boundary = closest quartile to 0
    nb = np.abs(h0) < max(eps, 1e-6)
    flip = float((np.sign(h1[nb]) != np.sign(h0[nb])).mean()) if nb.sum() > 0 else float("nan")
    dn = np.linalg.norm(D, axis=1) + 1e-12
    gn = np.linalg.norm(g0, axis=1) + 1e-12
    cos = (D * g0).sum(1) / (dn * gn)
    cosabs = float(np.abs(cos).mean()); cosmean = float(cos.mean())
    cka = linear_cka(Z0, Z1)
    # signal (safe<->unsafe) vs nuisance (red<->purple) — head-free
    d_appear = float(np.linalg.norm(mu))                       # inter-group (appearance) distance
    signuis = classgap / (d_appear + 1e-12)                    # >1 = label axis dominates appearance
    la = unsafe_m - safe_m
    cos_la = float(abs((mu @ la) / (np.linalg.norm(mu) * np.linalg.norm(la) + 1e-12)))  # appearance ∥ label?
    safe = (y == 0).astype(int)
    zs_acc = float(((h1 >= 0) == (safe == 1)).mean())
    zs_auc = float(roc_auc_score(safe, h1)) if 0 < safe.mean() < 1 else float("nan")
    return dict(displ=displ, transl=transl, deform=deform, displ_cg=displ_cg, cossim=cossim,
                nn1=nn1, medrank=medrank, dlabel=classgap, dappear=d_appear, signuis=signuis,
                cos_la=cos_la, corr=corr, dbias=dbias, flip=flip, cosabs=cosabs, cosmean=cosmean,
                cka=cka, zs_acc=zs_acc, zs_auc=zs_auc)


def make_states(seed, ec):
    rng = np.random.default_rng(seed); xs = ec.world_bounds
    return np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                     rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)


def run_lewm(enc, margin, axis, seed):
    from latent_cbf.adapters import LEWMWorldModel
    cfg = DreamerConfig(); cfg.lewm_ckpt_path = enc
    wm = LEWMWorldModel(cfg, enc).to(DEV)
    sd = torch.load(margin, map_location=DEV)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"]); wm.eval()
    ec = Config().environment; obst = ec.get_obstacles_list()
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=obst, goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    st = make_states(seed, ec)
    new = AXES[axis]
    Z0 = jepa_feats(wm, jepa_render(env, st, CONDS["red"])).cpu().numpy()
    Z1 = jepa_feats(wm, jepa_render(env, st, CONDS[new])).cpu().numpy()
    y = inside_obstacle(st[:, :2], CONDS[new]["shape"], obst)
    h0, g0 = head_and_grad(wm.heads["margin_gp"], Z0)
    h1, _ = head_and_grad(wm.heads["margin_gp"], Z1)
    return metrics(Z0, Z1, h0, g0, h1, y), Z0, Z1, y, wm.heads["margin_gp"]


def run_dreamer(rssm, axis, seed):
    from compare_critic_to_hj_dreamer import build_dreamer_wm
    from ood_margin_gp_dreamer import make_env, render_cond as dr_render, encode_feats, OBSTACLES
    cfg = DreamerConfig(); ec = Config().environment
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]
    cfg.size = [128, 128]; cfg.filter_mode = "cbf"; cfg.no_gp = False; cfg.wm_backend = "dreamer"
    wm = build_dreamer_wm(cfg, rssm, DEV)
    env = make_env(); st = make_states(seed, ec); new = AXES[axis]
    Z0 = encode_feats(wm, dr_render(env, st, CONDS["red"]), st[:, 2]).cpu().numpy()
    Z1 = encode_feats(wm, dr_render(env, st, CONDS[new]), st[:, 2]).cpu().numpy()
    y = inside_obstacle(st[:, :2], CONDS[new]["shape"], OBSTACLES)
    h0, g0 = head_and_grad(wm.heads["margin_gp"], Z0)
    h1, _ = head_and_grad(wm.heads["margin_gp"], Z1)
    return metrics(Z0, Z1, h0, g0, h1, y), Z0, Z1, y, wm.heads["margin_gp"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", required=True, choices=["lewm", "dreamer"])
    ap.add_argument("--enc"); ap.add_argument("--margin"); ap.add_argument("--rssm")
    ap.add_argument("--tag", required=True); ap.add_argument("--axis", required=True, choices=list(AXES))
    ap.add_argument("--seed", type=int, default=0); ap.add_argument("--dump")
    a = ap.parse_args()
    m, Z0, Z1, y, _head = run_lewm(a.enc, a.margin, a.axis, a.seed) if a.backend == "lewm" else run_dreamer(a.rssm, a.axis, a.seed)
    print(f"KEY probe {a.tag} axis={a.axis} " + " ".join(f"{k}={v:.3f}" for k, v in m.items()))
    if a.dump:
        Zc = np.vstack([Z0, Z1]); Zc = Zc - Zc.mean(0)
        _, _, Vt = np.linalg.svd(Zc, full_matrices=False)
        P = Zc @ Vt[:2].T                                  # 2D PCA on pooled red+purple
        n = Z0.shape[0]
        Path(a.dump).parent.mkdir(parents=True, exist_ok=True)
        np.savez(a.dump, xy=P, appear=np.r_[np.zeros(n), np.ones(n)], label=np.r_[y, y])


if __name__ == "__main__":
    main()
