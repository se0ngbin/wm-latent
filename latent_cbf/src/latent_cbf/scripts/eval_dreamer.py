import argparse
import functools
import os
import pathlib
import sys
from matplotlib.widgets import EllipseSelector
import numpy as np
import ruamel.yaml as yaml


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3_torch import models
from dreamerv3_torch import tools

import torch
from torch import nn
from torch import distributions as torchd
import collections

from tqdm import trange
from termcolor import cprint
import matplotlib.pyplot as plt
import gym
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io
from pathlib import Path
to_np = lambda x: x.detach().cpu().numpy()
from PyHJ.data import Collector, VectorReplayBuffer, BehaviorCollector
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs
from gymnasium import spaces
from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy
from PyHJ.data import Batch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from configs import DreamerConfig, Config
from configs.paths import RSSM_CHECKPOINT, TRAJS_DIR
# import Dreamer from scripts/dreamer_offline.py
from dreamer_offline import Dreamer

import h5py
import imageio

def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # step in logger is environmental step
    step = 0
    logger = tools.DebugLogger(logdir, config.action_repeat * step)

    print("Create envs.")
    
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate, shape=(1,), dtype=np.float32
    )
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    image_size = config.size[0] #128
    image_observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )

    
    obs_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(2,), dtype=np.float32
    )
    observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space
        })


    print("Action Space", action_space)
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]


    print("Simulate agent.")
    if getattr(config, "wm_backend", "dreamer") == "lewm":
        from latent_cbf.adapters import LEWMWorldModel
        ckpt = config.lewm_ckpt_path or config.lewm_run_name
        assert ckpt, "wm_backend=lewm requires lewm_ckpt_path or lewm_run_name"
        wm = LEWMWorldModel(config, ckpt).to(config.device)
        if config.lewm_margin_ckpt:
            margin_sd = torch.load(config.lewm_margin_ckpt, map_location=config.device)
            wm.heads["margin_gp"].load_state_dict(margin_sd["margin_gp"])
            wm.heads["margin_nogp"].load_state_dict(margin_sd["margin_nogp"])
        wm.eval()

        class _AgentShim:
            pass
        agent = _AgentShim()
        agent._wm = wm
    else:
        agent = Dreamer(
            observation_space,
            action_space,
            config,
            logger,
            None,
        ).to(config.device)
        filepath = str(RSSM_CHECKPOINT)
        agent.load_state_dict(torch.load(filepath)['agent_state_dict'])




    config.state_shape = (1,1,544,)
    config.action_shape = (1,)
    config.max_action = 1

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU
    critic_net = Net(
        config.state_shape,
        config.action_shape,
        hidden_sizes=config.critic_net,
        activation=critic_activation,
        concat=True,
        device=config.device
    )
    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay_pyhj)
    env_action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32) # joint action space
    actor_net = Net(config.state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
    actor = Actor(
        actor_net, config.action_shape, max_action=config.max_action, device=config.device
    ).to(config.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)

    policy_gp = DDPGPolicy(
    critic,
    critic_optim,
    tau=config.tau,
    gamma=config.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=config.exploration_noise),
    reward_normalization=config.rew_norm,
    estimation_step=config.n_step,
    action_space=env_action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=config.actor_gradient_steps,
    )

    policy_nogp = DDPGPolicy(
    critic,
    critic_optim,
    tau=config.tau,
    gamma=config.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=config.exploration_noise),
    reward_normalization=config.rew_norm,
    estimation_step=config.n_step,
    action_space=env_action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=config.actor_gradient_steps,
    )
    policy_gp_ckpt = torch.load(config.filter_directory_gp)
    policy_nogp_ckpt = torch.load(config.filter_directory_nogp)
    policy_gp.load_state_dict(policy_gp_ckpt)
    policy_nogp.load_state_dict(policy_nogp_ckpt)
    def evaluate_V(state, policy):
        tmp_obs = np.array(state)
        tmp_batch = Batch(obs = tmp_obs, info = Batch())
        tmp = policy.critic_old(tmp_batch.obs, policy(tmp_batch, model="actor_old").act)
        return tmp.cpu().detach().numpy().flatten()
    if config.filter_mode == 'cbf':
        traj_filepath = str(TRAJS_DIR / "wm_test_cbf.h5")
    elif config.filter_mode == 'lr':
        traj_filepath = str(TRAJS_DIR / "wm_test_lr.h5")
    else:
        traj_filepath = str(TRAJS_DIR / "wm_test.h5")

    # Aggregate per-trajectory success/violation metrics. Violation = any
    # failure step. Success requires goal_state on the h5 (PushT: block pose
    # within eps of goal) — when absent (current dubins test files) we fall
    # back to "no collision" as a proxy success signal.
    metrics_out = getattr(config, "results_out", None)
    if metrics_out:
        import json
        n_traj = 0
        viols = []
        succs = []
        with h5py.File(traj_filepath, 'r') as f:
            trajs = f['trajectories']
            goal = f.attrs.get('goal_state', None)
            eps_pos = float(f.attrs.get('eps_pos', 20.0))
            eps_ang = float(f.attrs.get('eps_ang', np.pi / 9))
            for tname in trajs.keys():
                traj = trajs[tname]
                failures = traj['failures'][:]
                states = traj['states'][:]
                viol = bool(np.any(failures > 0))
                viols.append(viol)
                tgoal = traj.attrs.get('goal_state', goal)
                if tgoal is not None and states.shape[-1] >= 5:
                    bx, by, ba = states[-1, 2], states[-1, 3], states[-1, 4]
                    pos_diff = float(np.hypot(bx - tgoal[0], by - tgoal[1]))
                    ang_diff = float(np.abs((ba - tgoal[2] + np.pi) % (2 * np.pi) - np.pi))
                    succ = (pos_diff < eps_pos) and (ang_diff < eps_ang) and (not viol)
                else:
                    succ = not viol
                succs.append(bool(succ))
                n_traj += 1
        out = {
            "n_trajectories": n_traj,
            "violation_rate": float(np.mean(viols)) if viols else 0.0,
            "success_rate": float(np.mean(succs)) if succs else 0.0,
            "filter_mode": config.filter_mode,
            "wm_backend": getattr(config, "wm_backend", "dreamer"),
            "lewm_run_name": getattr(config, "lewm_run_name", ""),
        }
        Path(metrics_out).parent.mkdir(parents=True, exist_ok=True)
        with open(metrics_out, "w") as fout:
            json.dump(out, fout, indent=2)
        print(f"wrote metrics -> {metrics_out}: {out}")

    chunk = 8
    hist = 5
    action_vis = True
    margin_vis = False
    if action_vis:
        with h5py.File(traj_filepath, 'r') as f:
            trajs = f['trajectories']
            for traj in trajs.keys():
            
                obs = trajs[traj]['observations'][:]
                actions = trajs[traj]['actions'][:]/2
                states = trajs[traj]['states'][:]
                failures = trajs[traj]['failures'][:]
                states_inp = np.concatenate([np.cos(states[:, [1]]), np.sin(states[:, [-1]])], axis=-1)
                N = obs.shape[0]

                images = []

                for n in range(N-5):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    
                    ax1.imshow(obs[n])
                    ax1.set_title(f'Frame {n}')
                    ax1.axis('off')
                    
                    # Right plot: Margins and failures
                    ax2.plot(actions[:n], 'g-', label='actions', linewidth=1)
                    ax2.plot(failures[:n], 'r-', label='failures', linewidth=1)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.legend()
                    ax2.set_title('Actions')
                    ax2.set_xlabel('Time Step')
                    ax2.set_ylabel('Value')
                    

                    ax2.set_ylim(-1.1, 1.1)
                    ax2.set_xlim(0, N)
                    
                    plt.tight_layout()

                    # Save to in-memory buffer instead of file
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Read from buffer and convert to numpy array
                    pil_image = Image.open(buffer)
                    pil_image = pil_image.resize((1112, 490), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_image)
                    images.append(img_array)
                    
                    plt.close()
                    buffer.close()
                    
                # Create GIF from in-memory images
                if images:
                    imageio.mimsave(f'visualizations/{config.filter_mode}_vis_{traj}.gif', images, fps=10)
                    print(f"GIF saved as visualizations/{config.filter_mode}_vis_{traj}.gif")
                else:
                    print("No images created")
    if margin_vis:
        with h5py.File(traj_filepath, 'r') as f:
            trajs = f['trajectories']
            for traj in trajs.keys():
            
                obs = trajs[traj]['observations'][:]
                actions = trajs[traj]['actions'][:]
                states = trajs[traj]['states'][:]
                failures = trajs[traj]['failures'][:]
                states_inp = np.concatenate([np.cos(states[:, [1]]), np.sin(states[:, [-1]])], axis=-1)

                margin_gps = []
                margin_nogps = []
                values_gp = []
                values_nogp = []
                total_steps = obs.shape[0]
                total_itr = int(total_steps / chunk)
                for i in range(total_itr-1):
                    idx = i*chunk
                    is_first = torch.zeros(hist)
                    is_first[0] = 1
                    obs_batch = {
                        'image': obs[None, idx:idx + hist],
                        'obs_state': states_inp[None, idx:idx+hist],
                        'action': actions[None, idx:idx+hist, None],
                        'is_first': is_first[None, :],
                        'is_terminal': torch.zeros(1, hist)

                    }
                    data = agent._wm.preprocess(obs_batch)
                    for k, v in data.items():
                        print(k, v.shape)
                    embed = agent._wm.encoder(data)
                    states, _ = agent._wm.dynamics.observe(
                    embed, data["action"], data["is_first"]
                    )
                    recon = agent._wm.heads["decoder"](agent._wm.dynamics.get_feat(states))["image"].mode()
                    print('recon', recon.shape)
                    init = {k: v[:, -1] for k, v in states.items()}
                    action_chunk = actions[None, idx+hist:idx+hist+chunk, None]
                    action_chunk = torch.tensor(action_chunk, device=config.device, dtype=torch.float32)
                    prior = agent._wm.dynamics.imagine_with_action(action_chunk, init)
                    margin_gp = agent._wm.heads["margin_gp"](agent._wm.dynamics.get_feat(prior)).squeeze().detach().cpu().numpy()
                    margin_nogp = agent._wm.heads["margin_nogp"](agent._wm.dynamics.get_feat(prior)).squeeze().detach().cpu().numpy()
                    feat_detached = agent._wm.dynamics.get_feat(prior).detach().cpu().numpy().squeeze()
                    print('feat_detached', feat_detached.shape)
                    values_gp.append(evaluate_V(feat_detached, policy_gp))
                    values_nogp.append(evaluate_V(feat_detached, policy_nogp))
                    margin_gps.append(margin_gp)
                    margin_nogps.append(margin_nogp)


                # make gif with observations and predictions
                margin_gp = np.array(margin_gps).flatten()
                margin_nogp = np.array(margin_nogps).flatten()
                values_gp = np.array(values_gp).flatten()
                values_nogp = np.array(values_nogp).flatten()
                N = obs.shape[0]
                failures = failures[hist:]
                obs = obs[hist:]
                images = []
                for n in range(N-5):
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                    
                    
                    ax1.imshow(obs[n])
                    ax1.set_title(f'Frame {n}')
                    ax1.axis('off')
                    
                    # Right plot: Margins and failures
                    ax2.plot(np.tanh(margin_gp[:n+1]), 'g-', label='Margin GP', linewidth=1)
                    ax2.plot(np.tanh(margin_nogp[:n+1]), 'orange', label='Margin NoGP', linewidth=1)
                    ax2.plot(values_gp[:n+1], 'g', linestyle='--',label='Values GP', linewidth=1)
                    ax2.plot(values_nogp[:n+1], 'orange', linestyle='--',label='Values NoGP', linewidth=1)
                    ax2.plot(failures[:n+1], 'r-', label='Failures', linewidth=1)
                    ax2.axhline(y=0, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=8, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=16, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=24, color='black', linestyle='--', alpha=0.5)
                    ax2.axvline(x=32, color='black', linestyle='--', alpha=0.5)
                    ax2.legend()
                    ax2.set_title('WM Predictions vs Ground Truth')
                    ax2.set_xlabel('Time Step')
                    ax2.set_ylabel('Value')
                    ax2.grid(True, alpha=0.3)
                    
                    # Set consistent y-axis limits
                    all_values = np.concatenate([margin_gp[:n+1], margin_nogp[:n+1], failures[:n+1]])
                    if len(all_values) > 0:
                        y_min, y_max = np.min(all_values), np.max(all_values)
                        y_range = y_max - y_min
                        if y_range > 0:
                            ax2.set_ylim(-1.1, 1.1)
                        ax2.set_xlim(0, N)
                    
                    plt.tight_layout()

                    # Save to in-memory buffer instead of file
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Read from buffer and convert to numpy array
                    pil_image = Image.open(buffer)
                    pil_image = pil_image.resize((1112, 490), Image.Resampling.LANCZOS)
                    img_array = np.array(pil_image)
                    images.append(img_array)
                    
                    plt.close()
                    buffer.close()
                    
                # Create GIF from in-memory images
                if images:
                    imageio.mimsave(f'visualizations/margin_gps_{traj}.gif', images, fps=10)
                    print(f"GIF saved as visualizations/margin_gps_{traj}.gif")
                else:
                    print("No images created")
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--filter_mode', type=str, default='none', choices=['none', 'cbf', 'lr'],
                       help='Filter mode to use for WM prediction')
    parser.add_argument('--no_gp', action='store_true', default=False, help='Use no GP')
    parser.add_argument('--wm_backend', type=str, default=None, choices=[None, 'dreamer', 'lewm'])
    parser.add_argument('--lewm_run_name', type=str, default=None)
    parser.add_argument('--lewm_ckpt_path', type=str, default=None)
    parser.add_argument('--lewm_margin_ckpt', type=str, default=None)
    parser.add_argument('--results_out', type=str, default=None,
                        help='Optional path to dump aggregated metrics as JSON.')
    cli_args = parser.parse_args()
    args = DreamerConfig()
    args.filter_mode = cli_args.filter_mode
    args.no_gp = cli_args.no_gp
    if cli_args.wm_backend is not None:
        args.wm_backend = cli_args.wm_backend
    if cli_args.lewm_run_name is not None:
        args.lewm_run_name = cli_args.lewm_run_name
    if cli_args.lewm_ckpt_path is not None:
        args.lewm_ckpt_path = cli_args.lewm_ckpt_path
    if cli_args.lewm_margin_ckpt is not None:
        args.lewm_margin_ckpt = cli_args.lewm_margin_ckpt
    args.results_out = cli_args.results_out
    env_conf = Config()

    args.turnRate = env_conf.max_angular_velocity
    args.x_min = env_conf.environment.world_bounds[0]
    args.x_max = env_conf.environment.world_bounds[1]
    args.y_min = env_conf.environment.world_bounds[2]
    args.y_max = env_conf.environment.world_bounds[3]
    args.size = env_conf.environment.image_size
    
    
    main(args)
