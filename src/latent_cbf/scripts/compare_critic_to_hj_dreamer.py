"""Dreamer analog of compare_critic_to_hj.py.

Compares a Dreamer-RSSM DDPG critic's latent value V_latent(x, y, θ) and the
margin head to the HJ ground-truth V_truth(x, y, θ) on the 3D Dubins grid.

Mirrors the feat path used at filter time in
controllers/diffusion_controller.py (FilteredDiffusionController):
  preprocess -> encoder -> dynamics.observe(single frame, is_first=1) -> get_feat.
The critic is rebuilt with norm_layer=LayerNorm to match wm_ddpg.py training.
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

import gym as old_gym  # noqa: E402,F401
import gymnasium as gym  # noqa: E402
from gymnasium import spaces as gspaces  # noqa: E402

from configs import DreamerConfig, Config  # noqa: E402
from latent_cbf.dubins.dubins_env import DubinsEnv  # noqa: E402
from dreamerv3_torch.models import WorldModel  # noqa: E402

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


def build_dreamer_wm(config, rssm_ckpt, device):
    """Mirror diffusion_controller.init_wm (dreamer branch)."""
    action_space = gym.spaces.Box(low=-config.turnRate, high=config.turnRate,
                                  shape=(1,), dtype=np.float32)
    low = np.array([config.x_min, config.y_min, -np.pi])
    high = np.array([config.x_max, config.y_max, np.pi])
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_obs = gym.spaces.Box(np.float32(midpoint - interval / 2),
                            np.float32(midpoint + interval / 2))
    image_size = config.size[0] if hasattr(config, "size") else 128
    image_obs = gym.spaces.Box(low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8)
    obs_obs = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
    observation_space = gym.spaces.Dict({"state": gt_obs, "obs_state": obs_obs, "image": image_obs})
    config.num_actions = action_space.shape[0]

    wm = WorldModel(observation_space, action_space, 0, config).to(device)
    wm.eval()
    ckpt = torch.load(rssm_ckpt, map_location=device)
    if "agent_state_dict" in ckpt:
        agent_state = ckpt["agent_state_dict"]
        wm_state = {k[14:]: v for k, v in agent_state.items() if k.startswith("_wm.")}
        wm.load_state_dict(wm_state)
    else:
        wm.load_state_dict(ckpt)
    return wm


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
    batch = Batch(obs=feats, info=Batch())
    act = policy(batch, model="actor_old").act
    v = policy.critic_old(feats, act)
    return v.view(-1).detach().cpu().numpy()


def main(args):
    device = "cuda:0"

    gt = np.load(args.gt)
    V_truth = gt["V"]
    xs = gt["grid_xs"]; ys = gt["grid_ys"]; ths = gt["grid_thetas"]
    nx, ny, nth = V_truth.shape
    print(f"GT grid: {nx}×{ny}×{nth}; safe-set fraction = {(V_truth >= 0).mean():.3f}")

    sx = max(nx // args.nx, 1); sy = max(ny // args.ny, 1); sth = max(nth // args.ntheta, 1)
    ix = np.arange(0, nx, sx); iy = np.arange(0, ny, sy); ith = np.arange(0, nth, sth)
    xs_s = xs[ix]; ys_s = ys[iy]; ths_s = ths[ith]
    Nx, Ny, Nth = len(xs_s), len(ys_s), len(ths_s)
    print(f"Eval grid: {Nx}×{Ny}×{Nth} = {Nx*Ny*Nth:,} cells")
    V_truth_s = V_truth[np.ix_(ix, iy, ith)]

    env_conf = Config()
    env = build_env(env_conf)

    config = DreamerConfig()
    # Populate runtime fields the controller normally sets (see collect_trajs.py).
    config.turnRate = env_conf.environment.max_angular_velocity
    config.x_min = env_conf.environment.world_bounds[0]
    config.x_max = env_conf.environment.world_bounds[1]
    config.y_min = env_conf.environment.world_bounds[2]
    config.y_max = env_conf.environment.world_bounds[3]
    config.size = [128, 128]
    config.filter_mode = "cbf"
    config.no_gp = False
    config.wm_backend = "dreamer"
    wm = build_dreamer_wm(config, args.rssm_ckpt, device)

    have_margin = f"margin_{args.margin_head}" in wm.heads

    # Render grid frames + record orientation.
    grid_imgs = []
    grid_ths = []
    print("Rendering grid frames...")
    for x in xs_s:
        for y in ys_s:
            for th in ths_s:
                env.state = np.array([x, y, th], dtype=np.float32)
                grid_imgs.append(env.render())
                grid_ths.append(float(th))
    grid_imgs = np.stack(grid_imgs, axis=0)
    grid_ths = np.asarray(grid_ths, dtype=np.float32)
    print(f"Got {grid_imgs.shape} images; encoding through Dreamer RSSM ...")

    # Determine feat size from one forward pass.
    def encode_feat(img_chunk, th_chunk):
        b = img_chunk.shape[0]
        obs_state = np.stack([np.cos(th_chunk), np.sin(th_chunk)], axis=-1)[:, None]  # (b,1,2)
        batch = {
            "image": img_chunk[:, None],                       # (b,1,H,W,3)
            "obs_state": obs_state.astype(np.float32),         # (b,1,2)
            "action": np.zeros((b, 1, 1), dtype=np.float32),
            "is_first": np.ones((b, 1, 1), dtype=np.float32),
            "is_terminal": np.zeros((b, 1, 1), dtype=np.float32),
        }
        data = wm.preprocess(batch)
        embed = wm.encoder(data)
        states, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        feat = wm.dynamics.get_feat(states)  # (b,1,544)
        return feat[:, -1]

    policy = None
    feat_size = None
    V_latent_flat = np.zeros(grid_imgs.shape[0], dtype=np.float32)
    margin_flat = np.zeros(grid_imgs.shape[0], dtype=np.float32)
    bs = args.batch_size
    for i in range(0, grid_imgs.shape[0], bs):
        chunk = grid_imgs[i:i + bs]
        th_chunk = grid_ths[i:i + bs]
        with torch.no_grad():
            feat = encode_feat(chunk, th_chunk)
            if policy is None:
                feat_size = int(feat.shape[-1])
                policy = load_ddpg(args.policy, feat_size, 1, device, config)
                print(f"feat_size={feat_size}; margin_head={'yes' if have_margin else 'no'}")
            V_latent_flat[i:i + bs] = eval_V_batch(policy, feat)
            if have_margin:
                m = torch.tanh(wm.heads[f"margin_{args.margin_head}"](feat)).reshape(-1).cpu().numpy()
                margin_flat[i:i + bs] = m

    V_latent = V_latent_flat.reshape(Nx, Ny, Nth)
    margin = margin_flat.reshape(Nx, Ny, Nth)

    # critic -> reachability V*; margin head (trained on per-step `failure`) ->
    # instantaneous constraint ℓ(x,y) = signed distance to obstacles (theta-indep).
    ell_s = np.full((Nx, Ny, 1), np.inf, dtype=np.float64)
    for cx, cy, r in gt["obstacles"]:
        ell_s = np.minimum(ell_s, np.sqrt((xs_s[:, None, None] - cx) ** 2
                                          + (ys_s[None, :, None] - cy) ** 2) - r)
    ell_s = np.broadcast_to(ell_s, (Nx, Ny, Nth))

    safe_truth = V_truth_s > 0
    safe_ell = ell_s > 0
    safe_latent = V_latent > 0
    safe_margin = margin > 0
    sign_agree_V = (safe_truth == safe_latent).mean()
    sign_agree_m = (safe_ell == safe_margin).mean() if have_margin else float("nan")

    try:
        from sklearn.metrics import roc_auc_score
        auc_V = float(roc_auc_score(safe_truth.flatten(), V_latent.flatten()))
        auc_m = float(roc_auc_score(safe_ell.flatten(), margin.flatten())) if have_margin else float("nan")
    except Exception:
        auc_V = float("nan"); auc_m = float("nan")

    print(f"sign-agree(V_latent vs V*) = {sign_agree_V:.3f}")
    print(f"sign-agree(margin   vs ℓ)  = {sign_agree_m:.3f}")
    print(f"AUC(V_latent vs V*)        = {auc_V:.3f}   [safe(V*)={safe_truth.mean():.3f}]")
    print(f"AUC(margin   vs ℓ)         = {auc_m:.3f}   [safe(ℓ)={safe_ell.mean():.3f}]")

    per_theta = []
    for k, th in enumerate(ths_s):
        per_theta.append({
            "theta": float(th),
            "sign_agree_V": float((safe_truth[:, :, k] == safe_latent[:, :, k]).mean()),
            "sign_agree_margin": (float((safe_ell[:, :, k] == safe_margin[:, :, k]).mean())
                                  if have_margin else float("nan")),
        })

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_dir / f"{args.tag}_V.npz",
                        V_latent=V_latent, margin=margin,
                        V_truth_subgrid=V_truth_s, ell_subgrid=ell_s,
                        xs=xs_s, ys=ys_s, thetas=ths_s)
    summary = {
        "run_name": args.tag,
        "grid": [Nx, Ny, Nth],
        "safe_frac_Vstar": float(safe_truth.mean()),
        "safe_frac_ell": float(safe_ell.mean()),
        "sign_agreement_V_vs_truth": float(sign_agree_V),     # critic vs V*
        "sign_agreement_margin_vs_ell": float(sign_agree_m),  # margin vs ℓ
        "auc_V_latent": auc_V,                                # vs V*
        "auc_margin": auc_m,                                  # vs ℓ
        "have_margin": bool(have_margin),
        "per_theta": per_theta,
    }
    with open(out_dir / f"{args.tag}_compare.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("done. summary:")
    print(json.dumps({k: v for k, v in summary.items() if k != "per_theta"}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt", default="/home/seongbin/latent/latent_cbf/results/hj_truth.npz")
    parser.add_argument("--rssm_ckpt", required=True)
    parser.add_argument("--policy", required=True, help="DDPG policy.pth path")
    parser.add_argument("--tag", required=True)
    parser.add_argument("--margin_head", default="gp", choices=["gp", "nogp"])
    parser.add_argument("--out_dir", default="/home/seongbin/latent/latent_cbf/results/hj_compare")
    parser.add_argument("--nx", type=int, default=27)
    parser.add_argument("--ny", type=int, default=27)
    parser.add_argument("--ntheta", type=int, default=16)
    parser.add_argument("--batch_size", type=int, default=64)
    main(parser.parse_args())
