"""Control-authority probe for the latent CBF filter.

§5 measured state-value fidelity: AUC of V_latent(s)=Q(s,π(s)) vs HJ V_truth(s).
But the deployed filter (controllers/diffusion_controller.py:481-502) freezes the
latent `feat` and only varies the ACTION input to the critic: it keeps actions with
Q(feat,a)/max_a Q(feat,a) >= cbf_gamma(=0.95) and overrides the nominal action only
if it's excluded. So all "control authority" is in the *shape of Q(feat,·) over a* —
a quantity the state-AUC never sees (it probes one action slice, a=π(s)).

This script sweeps Q(feat,a) over the 25-action filter grid at each Dubins grid state
and reports, on all states and on a boundary band (where filtering matters):
  M1 action-range   : (max-min)/|max| and /std(V)   — does Q vary with action at all
  M2 valid-fraction : mean[ Q/Qmax >= 0.95 ]         — ~1.0 => filter is a no-op
  M3 dir-agreement  : sign(argmax_a Q) vs sign(dV_truth/dθ)  — does it pick the right turn
plus a recomputed state-AUC (must match the §5 *_compare.json) as a regression check.

Modes:
  (default)     probe one WM -> <tag>_authority.{json,npz}
  --summarize   join all *_authority.json + §5 *_compare.json -> table + Spearman
  --curves T..  overlay Q(feat,a) vs a at shared boundary states for the given tags
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parent
_LATENT_CBF = _SCRIPTS.parent
_SRC = _LATENT_CBF.parent
_REPO_ROOT = _SRC.parent
for p in [str(_REPO_ROOT), str(_SRC), str(_LATENT_CBF), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.append(p)

DEVICE = "cuda:0"
GAMMA = 0.95          # cbf_gamma, configs/dreamer_conf.py:144
B_ACTIONS = 25        # filter's linspace(-1,1,25), diffusion_controller.py:482
OUT_DIR = Path("/home/seongbin/latent/latent_cbf/results/hj_compare")
GT_PATH = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"


# --------------------------------------------------------------------------- #
# grid + ground truth
# --------------------------------------------------------------------------- #
def load_gt_subgrid(nx, ny, ntheta):
    gt = np.load(GT_PATH)
    V = gt["V"]
    xs, ys, ths = gt["grid_xs"], gt["grid_ys"], gt["grid_thetas"]
    obstacles = gt["obstacles"]
    fnx, fny, fnth = V.shape
    sx, sy, sth = max(fnx // nx, 1), max(fny // ny, 1), max(fnth // ntheta, 1)
    ix, iy, ith = np.arange(0, fnx, sx), np.arange(0, fny, sy), np.arange(0, fnth, sth)
    # dV/dθ on the FULL periodic θ axis, then subsample (coarse θ ruins the derivative).
    dth = float(ths[1] - ths[0])
    dVdtheta_full = (np.roll(V, -1, axis=2) - np.roll(V, 1, axis=2)) / (2 * dth)
    sub = np.ix_(ix, iy, ith)
    return {
        "V": V[sub], "dVdtheta": dVdtheta_full[sub],
        "xs": xs[ix], "ys": ys[iy], "ths": ths[ith],
        "obstacles": obstacles,
    }


def boundary_mask(V_s, xs, ys, ths, obstacles, eps, d_obs):
    """Boundary = |V|<eps OR within d_obs of an obstacle surface. Shape (Nx,Ny,Nth)."""
    near_v = np.abs(V_s) < eps
    X, Y = np.meshgrid(xs, ys, indexing="ij")            # (Nx,Ny)
    sdf = np.full(X.shape, np.inf)
    for cx, cy, r in obstacles:
        sdf = np.minimum(sdf, np.sqrt((X - cx) ** 2 + (Y - cy) ** 2) - r)
    near_obs = (sdf < d_obs)[:, :, None] * np.ones((1, 1, len(ths)), dtype=bool)
    return near_v | near_obs


# --------------------------------------------------------------------------- #
# WM loading + feat encoding (mirrors the §5 compare scripts exactly)
# --------------------------------------------------------------------------- #
def render_grid(env, xs, ys, ths):
    imgs, th_flat = [], []
    for x in xs:
        for y in ys:
            for th in ths:
                env.state = np.array([x, y, th], dtype=np.float32)
                imgs.append(env.render())
                th_flat.append(float(th))
    return np.stack(imgs, 0), np.asarray(th_flat, dtype=np.float32)


def build_lewm(lewm_ckpt, margin_ckpt):
    from configs import DreamerConfig
    from latent_cbf.adapters import LEWMWorldModel
    config = DreamerConfig()
    config.lewm_ckpt_path = lewm_ckpt
    wm = LEWMWorldModel(config, lewm_ckpt).to(DEVICE)
    sd = torch.load(margin_ckpt, map_location=DEVICE)
    wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
    wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"])
    wm.eval()
    return wm, config


@torch.no_grad()
def encode_lewm(wm, imgs, bs=64):
    feats = []
    for i in range(0, imgs.shape[0], bs):
        chunk = imgs[i:i + bs][:, None]                  # (b,1,H,W,3)
        b = chunk.shape[0]
        batch = {"image": chunk,
                 "action": np.zeros((b, 1, 1), dtype=np.float32),
                 "is_first": np.ones((b, 1), dtype=np.float32),
                 "is_terminal": np.zeros((b, 1), dtype=np.float32)}
        data = wm.preprocess(batch)
        embed = wm.encoder(data)                          # (b,1,D)
        state = {"deter": embed[:, -1], "stoch": torch.zeros(b, 1, device=DEVICE)}
        feats.append(wm.dynamics.get_feat(state))         # (b,D)
    return torch.cat(feats, 0)


def build_dreamer(rssm_ckpt):
    from configs import DreamerConfig, Config
    import compare_critic_to_hj_dreamer as D
    env_conf = Config()
    config = DreamerConfig()
    config.turnRate = env_conf.environment.max_angular_velocity
    config.x_min = env_conf.environment.world_bounds[0]
    config.x_max = env_conf.environment.world_bounds[1]
    config.y_min = env_conf.environment.world_bounds[2]
    config.y_max = env_conf.environment.world_bounds[3]
    config.size = [128, 128]
    config.filter_mode = "cbf"; config.no_gp = False; config.wm_backend = "dreamer"
    wm = D.build_dreamer_wm(config, rssm_ckpt, DEVICE)
    return wm, config


@torch.no_grad()
def encode_dreamer(wm, imgs, ths, bs=64):
    feats = []
    for i in range(0, imgs.shape[0], bs):
        chunk = imgs[i:i + bs]; thc = ths[i:i + bs]; b = chunk.shape[0]
        obs_state = np.stack([np.cos(thc), np.sin(thc)], -1)[:, None].astype(np.float32)
        batch = {"image": chunk[:, None],
                 "obs_state": obs_state,
                 "action": np.zeros((b, 1, 1), dtype=np.float32),
                 "is_first": np.ones((b, 1, 1), dtype=np.float32),
                 "is_terminal": np.zeros((b, 1, 1), dtype=np.float32)}
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        states, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feats.append(wm.dynamics.get_feat(states)[:, -1])
    return torch.cat(feats, 0)


# --------------------------------------------------------------------------- #
# the action sweep + metrics
# --------------------------------------------------------------------------- #
@torch.no_grad()
def sweep_Q(policy, feats, B=B_ACTIONS, bs=4096):
    a = torch.linspace(-1, 1, B, device=feats.device)
    out = []
    for i in range(0, feats.shape[0], bs):
        f = feats[i:i + bs]
        n = f.shape[0]
        f_tiled = f.repeat_interleave(B, dim=0)          # (n*B,D)
        acts = a.repeat(n).unsqueeze(-1)                 # (n*B,1) cycles a per feat-block
        q = policy.critic_old(f_tiled, acts).view(n, B)
        out.append(q.cpu())
    return torch.cat(out, 0).numpy(), a.cpu().numpy()


def sign_convention_ok(env):
    """Empirically confirm normalized a>0 => θ̇>0 via the env's RK4 update."""
    env.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._update_state(np.array([env.max_angular_velocity], dtype=np.float32))
    up = float(env.state[2])
    env.state = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    env._update_state(np.array([-env.max_angular_velocity], dtype=np.float32))
    dn = float(env.state[2])
    return up > 0.0 > dn, up, dn


def compute_metrics(q, a, V_s, dVdtheta_s, bmask, dtheta_tol=1e-3):
    N, Bn = q.shape
    qmax = q.max(1); qmin = q.min(1)
    rng = qmax - qmin
    M1_rel = rng / (np.abs(qmax) + 1e-6)
    M1_vstd = rng / (V_s.std() + 1e-9)
    dec = q / qmax[:, None]                               # signed max, exactly as the filter
    M2 = (dec >= GAMMA).mean(1)
    a_best = a[q.argmax(1)]
    dvd = dVdtheta_s.reshape(-1)
    has_dir = np.abs(dvd) > dtheta_tol
    agree = (np.sign(a_best) == np.sign(dvd))

    bm = bmask.reshape(-1)
    def agg(m, mask):
        mask = mask & np.isfinite(m)
        return float(m[mask].mean()) if mask.any() else float("nan")
    res = {
        "M1_rel_all": agg(M1_rel, np.ones(N, bool)),
        "M1_rel_boundary": agg(M1_rel, bm),
        "M1_vstd_all": agg(M1_vstd, np.ones(N, bool)),
        "M1_vstd_boundary": agg(M1_vstd, bm),
        "M2_all": float(M2.mean()),
        "M2_boundary": agg(M2, bm),
        "M3_all": agg(agree.astype(float), has_dir),
        "M3_boundary": agg(agree.astype(float), bm & has_dir),
        "M3_n_boundary": int((bm & has_dir).sum()),
        "frac_qmax_neg": float((qmax < 0).mean()),
        "frac_qmax_neg_boundary": agg((qmax < 0).astype(float), bm),
        "n_boundary_cells": int(bm.sum()),
    }
    return res


# --------------------------------------------------------------------------- #
# per-WM probe
# --------------------------------------------------------------------------- #
def run_probe(args):
    import compare_critic_to_hj as Cl          # build_env, load_ddpg, eval_V_batch (lewm)
    from configs import Config
    gt = load_gt_subgrid(args.nx, args.ny, args.ntheta)
    xs, ys, ths = gt["xs"], gt["ys"], gt["ths"]
    Nx, Ny, Nth = len(xs), len(ys), len(ths)
    print(f"grid {Nx}x{Ny}x{Nth} = {Nx*Ny*Nth:,} cells")

    env = Cl.build_env(Config())
    ok, up, dn = sign_convention_ok(env)
    print(f"sign-convention check: a>0 => dθ={up:+.4f}, a<0 => dθ={dn:+.4f}  ({'OK' if ok else 'FLIPPED'})")
    assert ok, "action->theta sign convention is not as assumed; M3 would be inverted"

    imgs, th_flat = render_grid(env, xs, ys, ths)
    print(f"rendered {imgs.shape}; encoding ({args.backend}) ...")

    if args.backend == "lewm":
        wm, config = build_lewm(args.lewm_ckpt, args.margin_ckpt)
        feats = encode_lewm(wm, imgs)
        from compare_critic_to_hj import load_ddpg, eval_V_batch
    else:
        wm, config = build_dreamer(args.rssm_ckpt)
        feats = encode_dreamer(wm, imgs, th_flat)
        from compare_critic_to_hj_dreamer import load_ddpg, eval_V_batch
    feat_size = int(feats.shape[-1])
    policy = load_ddpg(args.policy, feat_size, 1, DEVICE, config)

    # state-AUC regression (must match §5)
    V_latent = eval_V_batch(policy, feats).reshape(Nx, Ny, Nth)
    safe_truth = (gt["V"] > 0).reshape(-1)
    try:
        from sklearn.metrics import roc_auc_score
        state_auc = float(roc_auc_score(safe_truth, V_latent.reshape(-1)))
    except Exception:
        state_auc = float("nan")

    q, a = sweep_Q(policy, feats)
    bmask = boundary_mask(gt["V"], xs, ys, ths, gt["obstacles"], args.eps, args.d_obs)
    metrics = compute_metrics(q, a, gt["V"], gt["dVdtheta"], bmask)

    summary = {"tag": args.tag, "backend": args.backend, "policy": args.policy,
               "grid": [Nx, Ny, Nth], "eps": args.eps, "d_obs": args.d_obs,
               "gamma": GAMMA, "B_actions": B_ACTIONS, "state_auc": state_auc,
               "sign_check": [up, dn], **metrics}
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_DIR / f"{args.tag}_authority.json", "w") as f:
        json.dump(summary, f, indent=2)
    # npz for the curves figure (WM-independent arrays shared across tags)
    np.savez_compressed(OUT_DIR / f"{args.tag}_authority.npz",
                        q=q.astype(np.float32), a=a.astype(np.float32),
                        V_s=gt["V"], dVdtheta=gt["dVdtheta"], bmask=bmask,
                        xs=xs, ys=ys, ths=ths, V_latent=V_latent.astype(np.float32))
    print(json.dumps(summary, indent=2))


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def run_summarize(_args):
    rows = []
    for jf in sorted(OUT_DIR.glob("*_authority.json")):
        a = json.load(open(jf))
        tag = a["tag"]
        cmpf = OUT_DIR / f"{tag}_compare.json"
        cmp_auc = json.load(open(cmpf))["auc_V_latent"] if cmpf.exists() else a.get("state_auc")
        rows.append((tag, a["backend"], cmp_auc, a["state_auc"],
                     a["M1_rel_boundary"], a["M2_all"], a["M2_boundary"],
                     a["M3_boundary"], a["frac_qmax_neg_boundary"], a["n_boundary_cells"]))
    hdr = f"{'tag':32s} {'be':7s} {'stAUC':>6s} {'M1relB':>7s} {'M2all':>6s} {'M2bnd':>6s} {'M3bnd':>6s} {'qnegB':>6s} {'nB':>5s}"
    print(hdr); print("-" * len(hdr))
    for (tag, be, auc5, _aucp, m1b, m2a, m2b, m3b, qn, nb) in rows:
        def f(v): return f"{v:.3f}" if isinstance(v, float) and np.isfinite(v) else " nan"
        print(f"{tag:32s} {be:7s} {f(auc5):>6s} {f(m1b):>7s} {f(m2a):>6s} {f(m2b):>6s} {f(m3b):>6s} {f(qn):>6s} {nb:5d}")
    print("\n(state-AUC vs safety) and (M2_boundary vs safety) Spearman: supply safety in --summarize via a mapping if desired.")


# --------------------------------------------------------------------------- #
# curves figure
# --------------------------------------------------------------------------- #
def run_curves(args):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    npzs = {t: np.load(OUT_DIR / f"{t}_authority.npz") for t in args.curves}
    ref = npzs[args.curves[0]]
    a = ref["a"]; bmask = ref["bmask"].reshape(-1); dvd = ref["dVdtheta"].reshape(-1)
    V_s = ref["V_s"].reshape(-1)
    # pick K boundary cells with clear preferred turn, spread by V
    cand = np.where(bmask & (np.abs(dvd) > 0.05))[0]
    cand = cand[np.argsort(V_s[cand])]
    K = min(args.k_curves, len(cand))
    pick = cand[np.linspace(0, len(cand) - 1, K).astype(int)]
    fig, axes = plt.subplots(1, K, figsize=(3.2 * K, 3.2), squeeze=False)
    for j, cell in enumerate(pick):
        ax = axes[0][j]
        for t in args.curves:
            q = npzs[t]["q"][cell]
            ax.plot(a, q / q.max(), marker=".", label=t)
        ax.axhline(GAMMA, ls="--", c="k", lw=0.8)
        astar = np.sign(dvd[cell])
        ax.axvline(astar, ls=":", c="green", lw=1.2)
        ax.set_title(f"V={V_s[cell]:+.2f} a*={astar:+.0f}", fontsize=8)
        ax.set_xlabel("action a"); ax.set_ylim(top=1.02)
        if j == 0:
            ax.set_ylabel("Q(feat,a)/maxQ"); ax.legend(fontsize=7)
    fig.suptitle("Control authority: normalized Q vs action at boundary states "
                 "(γ=0.95 dashed; HJ-optimal a* dotted)", fontsize=10)
    fig.tight_layout()
    out = OUT_DIR / "control_authority_curves.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"saved {out}")


# --------------------------------------------------------------------------- #
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["lewm", "dreamer"])
    p.add_argument("--lewm_ckpt"); p.add_argument("--margin_ckpt"); p.add_argument("--rssm_ckpt")
    p.add_argument("--policy"); p.add_argument("--tag")
    p.add_argument("--nx", type=int, default=41); p.add_argument("--ny", type=int, default=41)
    p.add_argument("--ntheta", type=int, default=32)
    p.add_argument("--eps", type=float, default=0.15); p.add_argument("--d_obs", type=float, default=0.25)
    p.add_argument("--summarize", action="store_true")
    p.add_argument("--curves", nargs="*", default=None)
    p.add_argument("--k_curves", type=int, default=6)
    args = p.parse_args()
    if args.summarize:
        run_summarize(args)
    elif args.curves:
        run_curves(args)
    else:
        run_probe(args)


if __name__ == "__main__":
    main()
