"""Compare a DDPG critic's latent value function V_latent(x, y, θ) to the HJ
ground-truth V_truth(x, y, θ) on the 3D Dubins state grid.

For each gridpoint:
  1. Place the Dubins env at state (x, y, θ); render the agent image.
  2. Encode via LE-WM (or Dreamer); pass through wm.dynamics.observe with a
     single-frame "history" to get the post latent.
  3. Query the loaded DDPG policy:  V(s) = critic_old(feat, actor_old(feat)).
  4. Compare to V_truth at the same gridpoint.

Reports:
  - Sign-agreement (safe-set match) across the grid.
  - ROC AUC for V_latent ranking V_truth>0 vs V_truth<0.
  - Per-θ-slice agreement (so you can see which orientations the critic gets right).
  - Saves V_latent.npy + a JSON summary; produces a 2D-slice PNG at θ=0.
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

from configs import DreamerConfig, Config  # noqa: E402
from latent_cbf.adapters import LEWMWorldModel  # noqa: E402
from latent_cbf.dubins.dubins_env import DubinsEnv  # noqa: E402

import gym as old_gym  # noqa: E402
import gymnasium as gym  # noqa: E402
from gymnasium import spaces as gspaces  # noqa: E402

from PyHJ.utils.net.common import Net  # noqa: E402
from PyHJ.utils.net.continuous import Actor, Critic  # noqa: E402
from PyHJ.exploration import GaussianNoise  # noqa: E402
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy  # noqa: E402
from PyHJ.data import Batch  # noqa: E402


def build_env(env_conf):
    obstacles = env_conf.environment.get_obstacles_list()
    env = DubinsEnv(
        image_size=tuple(env_conf.environment.image_size),
        world_bounds=tuple(env_conf.environment.world_bounds),
        max_angular_velocity=env_conf.environment.max_angular_velocity,
        speed=env_conf.environment.speed,
        dt=env_conf.environment.dt,
        obstacles=obstacles,
        goal_radius=env_conf.environment.goal_radius,
        collision_radius=env_conf.environment.collision_radius,
        render_mode="rgb_array",
    )
    return env


def load_ddpg(policy_path, feat_size, action_dim, device, dreamer_config):
    state_shape = (feat_size,)
    action_shape = (action_dim,)
    critic_net = Net(state_shape, action_shape,
                      hidden_sizes=dreamer_config.critic_net,
                      norm_layer=torch.nn.LayerNorm,
                      activation=torch.nn.ReLU, concat=True, device=device)
    critic = Critic(critic_net, device=device).to(device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=dreamer_config.critic_lr)
    actor_net = Net(state_shape, hidden_sizes=dreamer_config.control_net,
                     activation=torch.nn.ReLU, device=device)
    actor = Actor(actor_net, action_shape, max_action=1.0, device=device).to(device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=dreamer_config.actor_lr)
    policy = DDPGPolicy(
        critic, critic_optim,
        tau=dreamer_config.tau, gamma=dreamer_config.gamma_pyhj,
        exploration_noise=GaussianNoise(sigma=dreamer_config.exploration_noise),
        reward_normalization=dreamer_config.rew_norm,
        estimation_step=dreamer_config.n_step,
        action_space=gspaces.Box(low=-1, high=1, shape=action_shape, dtype=np.float32),
        actor=actor, actor_optim=actor_optim,
        actor_gradient_steps=dreamer_config.actor_gradient_steps,
    )
    policy.load_state_dict(torch.load(policy_path, map_location=device))
    policy.eval()
    return policy


@torch.no_grad()
def eval_V_batch(policy, feats):
    """V(s) = critic_old(feat, actor_old(feat)).  feats: (N, D) torch tensor."""
    batch = Batch(obs=feats, info=Batch())
    act = policy(batch, model="actor_old").act
    v = policy.critic_old(feats, act)
    return v.view(-1).detach().cpu().numpy()


def main(args):
    device = "cuda:0"

    # Load ground truth.
    gt = np.load(args.gt)
    V_truth = gt["V"]                 # (nx, ny, ntheta)
    xs = gt["grid_xs"]; ys = gt["grid_ys"]; ths = gt["grid_thetas"]
    nx, ny, nth = V_truth.shape
    print(f"GT grid: {nx}×{ny}×{nth}; safe-set fraction = {(V_truth >= 0).mean():.3f}")

    # Optional subsampling for speed.
    sx = max(nx // args.nx, 1); sy = max(ny // args.ny, 1); sth = max(nth // args.ntheta, 1)
    ix = np.arange(0, nx, sx); iy = np.arange(0, ny, sy); ith = np.arange(0, nth, sth)
    xs_s = xs[ix]; ys_s = ys[iy]; ths_s = ths[ith]
    Nx, Ny, Nth = len(xs_s), len(ys_s), len(ths_s)
    print(f"Eval grid: {Nx}×{Ny}×{Nth} = {Nx*Ny*Nth:,} cells")
    V_truth_s = V_truth[np.ix_(ix, iy, ith)]

    # Env (for rendering frames).
    env_conf = Config()
    env = build_env(env_conf)
    img_shape = env.image_size  # (W, H) in env spec

    # WM.
    config = DreamerConfig()
    config.lewm_ckpt_path = args.lewm_ckpt
    wm = LEWMWorldModel(config, args.lewm_ckpt).to(device)
    if args.margin_ckpt:
        sd = torch.load(args.margin_ckpt, map_location=device)
        wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
        wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"])
    wm.eval()

    feat_size = int(wm.embed_dim)
    policy = load_ddpg(args.policy, feat_size, action_dim=1, device=device, dreamer_config=config)

    # Render images on the grid and encode in batches.
    grid_pts = []
    grid_imgs = []
    print("Rendering grid frames...")
    for x in xs_s:
        for y in ys_s:
            for th in ths_s:
                env.state = np.array([x, y, th], dtype=np.float32)
                img = env.render()  # (H, W, 3) uint8
                grid_pts.append((float(x), float(y), float(th)))
                grid_imgs.append(img)
    grid_imgs = np.stack(grid_imgs, axis=0)  # (N, H, W, 3)
    print(f"Got {grid_imgs.shape} images; encoding through LE-WM ...")

    # Encode in chunks: preprocess expects (B, T, H, W, 3); use T=1 here.
    V_latent_flat = np.zeros(grid_imgs.shape[0], dtype=np.float32)
    margin_flat = np.zeros(grid_imgs.shape[0], dtype=np.float32)
    bs = args.batch_size
    for i in range(0, grid_imgs.shape[0], bs):
        chunk = grid_imgs[i:i+bs][:, None]  # (b, 1, H, W, 3)
        batch = {
            "image": chunk,
            "action": np.zeros((chunk.shape[0], 1, 1), dtype=np.float32),
            "is_first": np.ones((chunk.shape[0], 1), dtype=np.float32),
            "is_terminal": np.zeros((chunk.shape[0], 1), dtype=np.float32),
        }
        with torch.no_grad():
            data = wm.preprocess(batch)
            embed = wm.encoder(data)  # (b, 1, D)
            # Build a "state" dict matching the LE-WM adapter convention for get_feat.
            B = embed.shape[0]
            state = {
                "deter": embed[:, -1],  # (b, D)
                "stoch": torch.zeros(B, 1, device=device),
            }
            feat = wm.dynamics.get_feat(state)  # (b, D) for LE-WM
            v = eval_V_batch(policy, feat)
            m = torch.tanh(wm.heads[f"margin_{args.margin_head}"](feat)).reshape(-1).cpu().numpy()
        V_latent_flat[i:i+bs] = v
        margin_flat[i:i+bs] = m

    V_latent = V_latent_flat.reshape(Nx, Ny, Nth)
    margin = margin_flat.reshape(Nx, Ny, Nth)

    # Ground truth for each head:
    #   critic V_latent -> HJ reachability value V*  (V_truth_s)
    #   margin head     -> instantaneous constraint ℓ(x,y) = signed distance to
    #                      obstacles (theta-independent). The margin is trained on
    #                      the per-step `failure` flag, so ℓ is its correct target,
    #                      NOT V* (which is a strictly harder, smaller safe set).
    ell_s = np.full((Nx, Ny, 1), np.inf, dtype=np.float64)
    for cx, cy, r in gt["obstacles"]:
        ell_s = np.minimum(ell_s, np.sqrt((xs_s[:, None, None] - cx) ** 2
                                          + (ys_s[None, :, None] - cy) ** 2) - r)
    ell_s = np.broadcast_to(ell_s, (Nx, Ny, Nth))

    # Compute metrics.
    safe_truth = V_truth_s > 0      # critic reference (reachability)
    safe_ell = ell_s > 0            # margin reference (instantaneous)
    safe_latent = V_latent > 0
    safe_margin = margin > 0
    sign_agree_V = (safe_truth == safe_latent).mean()
    sign_agree_m = (safe_ell == safe_margin).mean()

    # ROC AUC: critic ranks vs V*, margin ranks vs ℓ.
    try:
        from sklearn.metrics import roc_auc_score
        auc_V = float(roc_auc_score(safe_truth.flatten(), V_latent.flatten()))
        auc_m = float(roc_auc_score(safe_ell.flatten(), margin.flatten()))
    except Exception:
        auc_V = float("nan"); auc_m = float("nan")

    print(f"sign-agree(V_latent vs V*) = {sign_agree_V:.3f}")
    print(f"sign-agree(margin   vs ℓ)  = {sign_agree_m:.3f}")
    print(f"AUC(V_latent vs V*)        = {auc_V:.3f}   [safe(V*)={safe_truth.mean():.3f}]")
    print(f"AUC(margin   vs ℓ)         = {auc_m:.3f}   [safe(ℓ)={safe_ell.mean():.3f}]")

    # Per-θ-slice sign agreement (for diagnostics).
    per_theta = []
    for k, th in enumerate(ths_s):
        per_theta.append({
            "theta": float(th),
            "sign_agree_V": float((safe_truth[:, :, k] == safe_latent[:, :, k]).mean()),
            "sign_agree_margin": float((safe_ell[:, :, k] == safe_margin[:, :, k]).mean()),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out_dir / f"{args.tag}_V.npz",
        V_latent=V_latent, margin=margin,
        V_truth_subgrid=V_truth_s, ell_subgrid=ell_s,
        xs=xs_s, ys=ys_s, thetas=ths_s,
    )
    summary = {
        "run_name": args.tag,
        "grid": [Nx, Ny, Nth],
        "safe_frac_Vstar": float(safe_truth.mean()),
        "safe_frac_ell": float(safe_ell.mean()),
        "sign_agreement_V_vs_truth": float(sign_agree_V),     # critic vs V*
        "sign_agreement_margin_vs_ell": float(sign_agree_m),  # margin vs ℓ
        "auc_V_latent": auc_V,                                # vs V*
        "auc_margin": auc_m,                                  # vs ℓ
        "per_theta": per_theta,
    }
    with open(out_dir / f"{args.tag}_compare.json", "w") as f:
        json.dump(summary, f, indent=2)

    # 2D slice plot at theta closest to 0.
    try:
        import matplotlib.pyplot as plt
        k0 = int(np.argmin(np.abs(ths_s)))
        fig, axes = plt.subplots(1, 4, figsize=(20, 5))
        slices = [V_truth_s[:, :, k0], V_latent[:, :, k0],
                  margin[:, :, k0],
                  (safe_truth[:, :, k0].astype(np.int8) - safe_latent[:, :, k0].astype(np.int8))]
        titles = [f"V_truth (θ={ths_s[k0]:.2f})", "V_latent (DDPG critic)",
                  "tanh(margin_gp)", "safe_truth - safe_latent (sign)"]
        for ax, sl, t in zip(axes, slices, titles):
            im = ax.imshow(sl.T, origin="lower",
                           extent=(xs_s[0], xs_s[-1], ys_s[0], ys_s[-1]), cmap="RdBu_r")
            ax.set_title(t); ax.set_xlabel("x"); ax.set_ylabel("y")
            for cx, cy, r in gt["obstacles"]:
                ax.add_patch(plt.Circle((cx, cy), r, ec="k", fc="none", lw=1.5))
            fig.colorbar(im, ax=ax, shrink=0.8)
        plt.tight_layout()
        png = out_dir / f"{args.tag}_slice_theta0.png"
        plt.savefig(png, dpi=110, bbox_inches="tight")
        plt.close()
        print(f"plot -> {png}")
    except Exception as e:
        print(f"plot skipped: {e}")

    print("done. summary:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/home/seongbin/latent/latent_cbf/results/hj_truth.npz")
    parser.add_argument("--lewm_ckpt", required=True)
    parser.add_argument("--margin_ckpt", required=True)
    parser.add_argument("--policy", required=True, help="DDPG policy.pth path")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--margin_head", default="gp", choices=["gp", "nogp"])
    parser.add_argument("--out_dir", default="/home/seongbin/latent/latent_cbf/results/hj_compare")
    parser.add_argument("--nx", type=int, default=41)
    parser.add_argument("--ny", type=int, default=41)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    main(parser.parse_args())
