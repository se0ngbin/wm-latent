import argparse
import os
import sys
import pprint

import gymnasium #as gym
import gym
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
# Also make `latent_cbf` (src/latent_cbf parent) importable for adapters/.
_src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
if _src_dir not in sys.path:
    sys.path.append(_src_dir)
from dreamerv3_torch import models
from dreamerv3_torch import tools
import ruamel.yaml as yaml
import wandb
from PyHJ.data import Collector, VectorReplayBuffer, BehaviorCollector
from PyHJ.env import DummyVectorEnv
from PyHJ.exploration import GaussianNoise
from PyHJ.trainer import offpolicy_trainer
from PyHJ.utils import TensorboardLogger, WandbLogger
from PyHJ.utils.net.common import Net
from PyHJ.utils.net.continuous import Actor, Critic
import PyHJ.reach_rl_gym_envs as reach_rl_gym_envs

from termcolor import cprint
from datetime import datetime
import pathlib
from pathlib import Path
import collections
from PIL import Image
import io
from PyHJ.data import Batch
import matplotlib.pyplot as plt
from configs import DreamerConfig, Config, get_diffusion_config
from dreamer_offline import make_dataset





def _build_wm(env, config):
    """Construct the world model used by the CBF pipeline. Branches on
    ``config.wm_backend``: 'dreamer' uses the original RSSM checkpoint;
    'lewm' wraps a frozen LE-WM (JEPA) checkpoint via the adapter."""
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
        return wm

    wm = models.WorldModel(env.observation_space_full, env.action_space, 0, config)
    ckpt_path = config.rssm_ckpt_path
    checkpoint = torch.load(ckpt_path)
    state_dict = {k[14:]: v for k, v in checkpoint['agent_state_dict'].items() if '_wm' in k}
    wm.load_state_dict(state_dict)
    # Retrained value-function column: swap in per-appearance margin heads.
    if getattr(config, "dreamer_margin_ckpt", None):
        msd = torch.load(config.dreamer_margin_ckpt, map_location=config.device)
        wm.heads["margin_gp"].load_state_dict(msd["margin_gp"])
        wm.heads["margin_nogp"].load_state_dict(msd["margin_nogp"])
        print(f"[wm_ddpg] loaded dreamer OOD margin from {config.dreamer_margin_ckpt}")
    wm.eval()
    return wm


def main(exp_config, config):
    env = gymnasium.make(config.task, params = [config])
    config.num_actions = env.action_space.n if hasattr(env.action_space, "n") else env.action_space.shape[0]
    wm = _build_wm(env, config)

    from controllers.factory import create_controller_from_config
    dp = create_controller_from_config(exp_config)

    config.batch_size = 1
    config.batch_length = 3
    expert_eps = collections.OrderedDict()
    expert_val_eps = collections.OrderedDict()
    tools.fill_offline_dataset(config, expert_eps, expert_val_eps)
    offline_dataset = make_dataset(expert_eps, config)



    env.set_wm(wm, offline_dataset, config, dp)


    # check if the environment has control and disturbance actions:
    assert hasattr(env, 'action_space') #and hasattr(env, 'action2_space'), "The environment does not have control and disturbance actions!"
    config.state_shape = env.observation_space.shape or env.observation_space.n
    config.action_shape = env.action_space.shape or env.action_space.n
    config.max_action = env.action_space.high[0]



    train_envs = DummyVectorEnv(
        [lambda: gymnasium.make(config.task, params = [wm, offline_dataset, config, dp]) for _ in range(config.training_num)]
    )
    test_envs = DummyVectorEnv(
        [lambda: gymnasium.make(config.task, params = [wm, offline_dataset, config, dp]) for _ in range(config.test_num)]
    )


    # seed
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    train_envs.seed(config.seed)
    test_envs.seed(config.seed)
    # model

    actor_activation = torch.nn.ReLU
    critic_activation = torch.nn.ReLU

    if config.critic_net is not None:
        critic_net = Net(
            config.state_shape,
            config.action_shape,
            hidden_sizes=config.critic_net,
            norm_layer=torch.nn.LayerNorm,
            activation=critic_activation,
            concat=True,
            device=config.device
        )
    else:
        # report error:
        raise ValueError("Please provide critic_net!")

    critic = Critic(critic_net, device=config.device).to(config.device)
    critic_optim = torch.optim.AdamW(critic.parameters(), lr=config.critic_lr, weight_decay=config.weight_decay_pyhj)

    log_path = None

    from PyHJ.policy import avoid_DDPGPolicy_annealing as DDPGPolicy

    print("DDPG under the Avoid annealed Bellman equation with no Disturbance has been loaded!")

    actor_net = Net(config.state_shape, hidden_sizes=config.control_net, activation=actor_activation, device=config.device)
    actor = Actor(
        actor_net, config.action_shape, max_action=config.max_action, device=config.device
    ).to(config.device)
    actor_optim = torch.optim.AdamW(actor.parameters(), lr=config.actor_lr)


    policy = DDPGPolicy(
    critic,
    critic_optim,
    tau=config.tau,
    gamma=config.gamma_pyhj,
    exploration_noise=GaussianNoise(sigma=config.exploration_noise),
    reward_normalization=config.rew_norm,
    estimation_step=config.n_step,
    action_space=env.action_space,
    actor=actor,
    actor_optim=actor_optim,
    actor_gradient_steps=config.actor_gradient_steps,
    )

    if config.no_gp:
        log_path = os.path.join(config.logdir, 'PyHJ/nogp')
    else:
        log_path = os.path.join(config.logdir+'/PyHJ/gp')


    # collector
    train_collector = BehaviorCollector(
        policy,
        train_envs,
        VectorReplayBuffer(config.buffer_size, len(train_envs)),
        exploration_noise=True
    )
    test_collector = Collector(policy, test_envs)

    if config.warm_start_path is not None:
        policy.load_state_dict(torch.load(config.warm_start_path))
        config.kwargs = config.kwargs + "warmstarted"

    epoch = 0



    if config.continue_training_epoch is not None:
        epoch = config.continue_training_epoch
        policy.load_state_dict(torch.load(
            os.path.join(
                log_path+"/epoch_id_{}".format(epoch),
                "policy.pth"
            )
        ))


    if config.continue_training_logdir is not None:
        policy.load_state_dict(torch.load(config.continue_training_logdir))
        epoch = config.continue_training_epoch


    def save_best_fn(policy, epoch=epoch):
        target_dir = log_path + "/epoch_id_{}".format(epoch)
        os.makedirs(target_dir, exist_ok=True)
        torch.save(
            policy.state_dict(),
            os.path.join(target_dir, "policy.pth"),
        )


    def stop_fn(mean_rewards):
        return False


    if not os.path.exists(log_path+"/epoch_id_{}".format(epoch)):
        print("Just created the log directory!")
        # print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        os.makedirs(log_path+"/epoch_id_{}".format(epoch))

        
    logger = None
    warmup = 1

    for iter in range(warmup+config.total_episodes):
        if iter  < warmup:
            policy._gamma = 0 # for warmup the value fn
            policy.warmup = True
        else:
            policy._gamma = config.gamma_pyhj
            policy.warmup = False

        if config.continue_training_epoch is not None:
            print("epoch: {}, remaining epochs: {}".format(epoch//config.epoch, config.total_episodes - iter))
        else:
            print("epoch: {}, remaining epochs: {}".format(iter, config.total_episodes - iter))
        epoch = epoch + config.epoch
        print("log_path: ", log_path+"/epoch_id_{}".format(epoch))
        if config.total_episodes > 1:
            writer = SummaryWriter(log_path+"/epoch_id_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        else:
            if not os.path.exists(log_path+"/total_epochs_{}".format(epoch)):
                print("Just created the log directory!")
                print("log_path: ", log_path+"/total_epochs_{}".format(epoch))
                os.makedirs(log_path+"/total_epochs_{}".format(epoch))
            writer = SummaryWriter(log_path+"/total_epochs_{}".format(epoch)) #filename_suffix="_"+timestr+"_epoch_id_{}".format(epoch))
        if logger is None:
            logger = WandbLogger()
            logger.load(writer)
        logger = TensorboardLogger(writer)
        
        # import pdb; pdb.set_trace()
        result = offpolicy_trainer(
        policy,
        train_collector,
        test_collector,
        config.epoch,
        config.step_per_epoch,
        config.step_per_collect,
        config.test_num,
        config.batch_size_pyhj,
        update_per_step=config.update_per_step,
        stop_fn=stop_fn,
        save_best_fn=save_best_fn,
        logger=logger
        )
        
        save_best_fn(policy, epoch=epoch)
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no_gp", action="store_true", default=False)
    parser.add_argument("--wm_backend", type=str, default=None, choices=[None, "dreamer", "lewm"])
    parser.add_argument("--lewm_run_name", type=str, default=None)
    parser.add_argument("--lewm_ckpt_path", type=str, default=None)
    parser.add_argument("--lewm_margin_ckpt", type=str, default=None)
    parser.add_argument("--dreamer_margin_ckpt", type=str, default=None,
                        help="For wm_backend=dreamer: override the WM's margin heads "
                             "with per-appearance margins (retrained value-function column).")
    parser.add_argument("--buffer_path", type=str, default=None,
                        help="Override the dataset path used for env reset seeding.")
    parser.add_argument("--step_per_epoch", type=int, default=None)
    parser.add_argument("--total_episodes", type=int, default=None)
    parser.add_argument("--uniform_reset", action="store_true",
                        help="Sample env reset states uniformly over the safe region instead of from the offline buffer; gives the DDPG critic broader latent coverage.")
    parser.add_argument("--gamma_pyhj", type=float, default=None,
                        help="DDPG discount factor; default 0.9999 collapses the bootstrap to ≈ margin reward over short rollouts. Try 0.99 for actual Bellman propagation.")
    parser.add_argument("--reward_scale", type=float, default=None,
                        help="Multiply env reward (tanh'd margin) by this. Sharpens the Bellman signal.")
    parser.add_argument("--logdir", type=str, default=None,
                        help="Override DDPG output root. Default writes to data/dreamer/PyHJ/{gp,nogp}/... which collides across LE-WM variants; pass a per-variant dir.")
    parser.add_argument("--rssm-ckpt", type=str, default=None,
                        help="Override the world-model checkpoint path. Defaults to <logdir>/rssm_ckpt.pt when --logdir is set.")
    parser.add_argument("--seed", type=int, default=None,
                        help="PyHJ training seed (for multi-seed error bars). Use a per-seed --logdir to avoid clobbering.")
    args = parser.parse_args()
    exp_config = get_diffusion_config()
    config = DreamerConfig()
    if args.seed is not None:
        config.seed = args.seed
    config.no_gp = bool(args.no_gp)
    if args.wm_backend is not None:
        config.wm_backend = args.wm_backend
    if args.lewm_run_name is not None:
        config.lewm_run_name = args.lewm_run_name
    if args.lewm_ckpt_path is not None:
        config.lewm_ckpt_path = args.lewm_ckpt_path
    if args.lewm_margin_ckpt is not None:
        config.lewm_margin_ckpt = args.lewm_margin_ckpt
    config.dreamer_margin_ckpt = args.dreamer_margin_ckpt
    if args.buffer_path is not None:
        config.dataset_path = args.buffer_path
    if args.step_per_epoch is not None:
        config.step_per_epoch = args.step_per_epoch
    if args.total_episodes is not None:
        config.total_episodes = args.total_episodes
    if args.logdir is not None:
        config.logdir = args.logdir
        config.rssm_ckpt_path = os.path.join(args.logdir, "rssm_ckpt.pt")
    if args.rssm_ckpt is not None:
        config.rssm_ckpt_path = args.rssm_ckpt
    config.uniform_reset = bool(args.uniform_reset)
    if args.gamma_pyhj is not None:
        config.gamma_pyhj = args.gamma_pyhj
    if args.reward_scale is not None:
        config.reward_scale = args.reward_scale
    else:
        config.reward_scale = getattr(config, "reward_scale", 1.0)
    config.size = exp_config.environment.image_size
    config.turnRate = exp_config.max_angular_velocity
    main(exp_config, config)