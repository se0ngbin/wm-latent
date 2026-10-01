import os

os.environ["MUJOCO_GL"] = "egl"

import time
from pathlib import Path

import hydra
import numpy as np
import stable_pretraining as spt
import torch
from omegaconf import DictConfig, OmegaConf
from sklearn import preprocessing
from torchvision.transforms import v2 as transforms
import stable_worldmodel as swm
import dubins_swm_env  # noqa: F401  (registers swm/Dubins-v0)

def img_transform(cfg):
    transform = transforms.Compose(
        [
            transforms.ToImage(),
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(**spt.data.dataset_stats.ImageNet),
            transforms.Resize(size=cfg.eval.img_size),
        ]
    )
    return transform


def get_episodes_length(dataset, episodes):
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"

    episode_idx = dataset.get_col_data(col_name)
    step_idx = dataset.get_col_data("step_idx")
    lengths = []
    for ep_id in episodes:
        lengths.append(np.max(step_idx[episode_idx == ep_id]) + 1)
    return np.array(lengths)


def get_dataset(cfg, dataset_name):
    dataset_path = Path(cfg.cache_dir or swm.data.utils.get_cache_dir())
    dataset = swm.data.HDF5Dataset(
        dataset_name,
        keys_to_cache=cfg.dataset.keys_to_cache,
        cache_dir=dataset_path,
    )
    return dataset

@hydra.main(version_base=None, config_path="./config/eval", config_name="pusht")
def run(cfg: DictConfig):
    """Run evaluation of dinowm vs random policy."""
    assert (
        cfg.plan_config.horizon * cfg.plan_config.action_block <= cfg.eval.eval_budget
    ), "Planning horizon must be smaller than or equal to eval_budget"

    # create world environment
    cfg.world.max_episode_steps = 2 * cfg.eval.eval_budget
    image_shape = tuple(cfg.eval.get("world_image_shape", (224, 224)))
    world = swm.World(**cfg.world, image_shape=image_shape)

    # create the transform
    transform = {
        "pixels": img_transform(cfg),
        "goal": img_transform(cfg),
    }

    dataset = get_dataset(cfg, cfg.eval.dataset_name)
    stats_dataset = dataset  # get_dataset(cfg, cfg.dataset.stats)
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    ep_indices, _ = np.unique(stats_dataset.get_col_data(col_name), return_index=True)

    process = {}
    for col in cfg.dataset.keys_to_cache:
        if col in ["pixels"]:
            continue
        processor = preprocessing.StandardScaler()
        col_data = stats_dataset.get_col_data(col)
        col_data = col_data[~np.isnan(col_data).any(axis=1)]
        processor.fit(col_data)
        process[col] = processor

        if col != "action":
            process[f"goal_{col}"] = process[col]

    # -- run evaluation
    policy = cfg.get("policy", "random")

    if policy != "random":
        model = swm.wm.utils.load_pretrained(cfg.policy)
        model = model.to("cuda")
        model = model.eval()
        model.requires_grad_(False)
        model.interpolate_pos_encoding = True
        config = swm.PlanConfig(**cfg.plan_config)
        solver = hydra.utils.instantiate(cfg.solver, model=model)
        if hasattr(solver, "set_process"):  # safe_cem: needs dataset scalers
            solver.set_process(process)
        policy = swm.policy.WorldModelPolicy(
            solver=solver, config=config, process=process, transform=transform
        )

    else:
        policy = swm.policy.RandomPolicy()

    results_path = (
        Path(swm.data.utils.get_cache_dir(), cfg.policy).parent
        if cfg.policy != "random"
        else Path(__file__).parent
    )

    # sample the episodes and the starting indices
    episode_len = get_episodes_length(dataset, ep_indices)
    max_start_idx = episode_len - cfg.eval.goal_offset_steps - 1
    max_start_idx_dict = {ep_id: max_start_idx[i] for i, ep_id in enumerate(ep_indices)}
    # Map each dataset row’s episode_idx to its max_start_idx
    col_name = "episode_idx" if "episode_idx" in dataset.column_names else "ep_idx"
    max_start_per_row = np.array(
        [max_start_idx_dict[ep_id] for ep_id in dataset.get_col_data(col_name)]
    )

    # remove all the lines of dataset for which dataset['step_idx'] > max_start_per_row
    step_idx_col = dataset.get_col_data("step_idx")
    if cfg.eval.get("goal_at_episode_end", False):
        # Pin the goal to each episode's final frame: the only valid start per
        # episode is exactly goal_offset steps before the end. Keeps the goal
        # marker at the true target so rendered frames match the goal image.
        valid_mask = step_idx_col == max_start_per_row
    else:
        valid_mask = step_idx_col <= max_start_per_row
    valid_indices = np.nonzero(valid_mask)[0]

    # Optional: keep only starts whose goal (state at start+goal_offset) sits in
    # an annulus *around* an obstacle — outside the disk (reachable, not inside)
    # but close to it, so the task exercises obstacle avoidance. Keep goals whose
    # distance to the NEAREST obstacle edge is in (avoid_margin, near_margin].
    obstacles = cfg.eval.get("goal_avoid_obstacles", None)
    if obstacles and not cfg.eval.get("goal_crosses_obstacle", False):
        avoid = float(cfg.eval.get("goal_avoid_margin", 0.05))   # min clearance from edge
        near = float(cfg.eval.get("goal_near_margin", 0.5))       # max distance from edge
        state_col = dataset.get_col_data("state")
        goal_pos = state_col[valid_indices + cfg.eval.goal_offset_steps][:, :2]
        edge_dist = np.full(len(valid_indices), np.inf)
        for ox, oy, r in obstacles:
            edge_dist = np.minimum(
                edge_dist, np.linalg.norm(goal_pos - np.array([ox, oy]), axis=1) - r
            )
        keep = (edge_dist > avoid) & (edge_dist <= near)
        valid_indices = valid_indices[keep]
        print(int(keep.sum()), f"goals in annulus ({avoid}, {near}] around an obstacle")

    # Optional (far-side stress test): keep only starts whose STRAIGHT LINE to the
    # goal crosses an obstacle disk — so the greedy path goes *through* it and
    # avoidance requires a detour that opposes the goal-distance objective. Goal
    # itself must be outside all obstacles (reachable). Complements the annulus
    # filter (which puts goals *beside* an obstacle so precision == avoidance).
    if cfg.eval.get("goal_crosses_obstacle", False):
        obs = cfg.eval.get("goal_avoid_obstacles")
        state_col = dataset.get_col_data("state")
        p0 = state_col[valid_indices][:, :2]                              # start
        p1 = state_col[valid_indices + cfg.eval.goal_offset_steps][:, :2]  # goal
        crosses = np.zeros(len(valid_indices), bool)
        goal_outside = np.ones(len(valid_indices), bool)
        for ox, oy, r in obs:
            c = np.array([ox, oy]); d = p1 - p0; f = p0 - c
            t = np.clip(-(f * d).sum(1) / ((d * d).sum(1) + 1e-9), 0.0, 1.0)  # closest-point param
            seg_dist = np.linalg.norm(p0 + t[:, None] * d - c, axis=1)         # segment-to-center
            crosses |= seg_dist < r
            goal_outside &= np.linalg.norm(p1 - c, axis=1) > r
        keep = crosses & goal_outside
        valid_indices = valid_indices[keep]
        print(int(keep.sum()), "goals whose straight-line path crosses an obstacle (far-side)")

    # Optional (TwoRoom): keep only start/goal pairs in DIFFERENT rooms — i.e. on
    # opposite sides of the dividing wall — so reaching the goal requires threading
    # the door. Isolates the cases that actually exercise the wall. Reads a
    # configurable position column (proprio for TwoRoom) and wall coordinate.
    if cfg.eval.get("goal_cross_wall", False):
        col = cfg.eval.get("pos_col", "proprio")
        ax = int(cfg.eval.get("wall_axis_coord", 0))          # 0=x
        wpos = float(cfg.eval.get("wall_pos", 112.0))
        pos = dataset.get_col_data(col)
        s0 = pos[valid_indices][:, ax]
        s1 = pos[valid_indices + cfg.eval.goal_offset_steps][:, ax]
        keep = (s0 - wpos) * (s1 - wpos) < 0                  # opposite sides
        valid_indices = valid_indices[keep]
        print(int(keep.sum()), "start/goal pairs in different rooms (cross-wall)")

    # Optional: only draw goals from clean-SUCCESS source episodes — the source
    # trajectory reaches the goal region (end x>x_min, |y|<y_abs) AND never
    # collides. Makes the sub-goals genuine expert midpoints (pushT-comparable),
    # not midpoints of exploratory/crash rollouts in the mixed buffer.
    if cfg.eval.get("goal_clean_source", False):
        ep_col_all = dataset.get_col_data(col_name)
        fail = np.asarray(dataset.get_col_data("failures")).reshape(-1)
        st_all = dataset.get_col_data("state")
        gx = float(cfg.eval.get("clean_goal_x_min", 1.0))
        gy = float(cfg.eval.get("clean_goal_y_abs", 0.8))
        uniq, offs, lens = np.unique(ep_col_all, return_index=True, return_counts=True)
        order = np.argsort(offs)
        offs, lens, eids = offs[order], lens[order], uniq[order]
        end_st = st_all[offs + lens - 1]
        reaches = (end_st[:, 0] > gx) & (np.abs(end_st[:, 1]) < gy)
        ep_coll = np.add.reduceat((fail > 0).astype(np.int64), offs) > 0
        clean_map = dict(zip(eids.tolist(), (reaches & ~ep_coll).tolist()))
        keep = np.array([clean_map[int(ep_col_all[r])] for r in valid_indices])
        valid_indices = valid_indices[keep]
        print(int(keep.sum()), "goals from clean-success source episodes")

    print(len(valid_indices), "valid starting points found for evaluation.")

    g = np.random.default_rng(cfg.seed)
    random_episode_indices = g.choice(
        len(valid_indices) - 1, size=cfg.eval.num_eval, replace=False
    )

    # sort increasingly to avoid issues with HDF5Dataset indexing
    random_episode_indices = np.sort(valid_indices[random_episode_indices])

    print(random_episode_indices)

    eval_episodes = dataset.get_row_data(random_episode_indices)[col_name]
    eval_start_idx = dataset.get_row_data(random_episode_indices)["step_idx"]

    if len(eval_episodes) < cfg.eval.num_eval:
        raise ValueError("Not enough episodes with sufficient length for evaluation.")

    world.set_policy(policy)

    results_path.mkdir(parents=True, exist_ok=True)

    start_time = time.time()
    metrics = world.evaluate(
        dataset=dataset,
        start_steps=eval_start_idx.tolist(),
        goal_offset=cfg.eval.goal_offset_steps,
        eval_budget=cfg.eval.eval_budget,
        episodes_idx=eval_episodes.tolist(),
        callables=OmegaConf.to_container(cfg.eval.get("callables"), resolve=True),
        video=results_path,
    )
    end_time = time.time()

    # Per-episode collision rate for envs that expose it (e.g. Dubins). In
    # dataset-eval 'wait' mode each env runs exactly one episode, so the env's
    # `_collided` flag at the end is that episode's outcome.
    pool_envs = getattr(world.envs, "envs", None)
    if pool_envs is not None and all(
        hasattr(e.unwrapped, "_collided") for e in pool_envs
    ):
        collided = np.array([bool(e.unwrapped._collided) for e in pool_envs])
        metrics["collision_rate"] = float(collided.mean() * 100.0)
        metrics["episode_collisions"] = collided
        # 2x2 joint outcome breakdown: {reached goal} x {collided en route}. Uses the
        # env's own episode-level _reached flag (goal reached within success_tol) so it
        # is exactly consistent with the collision flag on the same trajectories.
        if all(hasattr(e.unwrapped, "_reached") for e in pool_envs):
            reached = np.array([bool(e.unwrapped._reached) for e in pool_envs])
            n = float(len(reached))
            metrics["safe_success"]   = float(((reached) & (~collided)).mean() * 100.0)  # reached, no collision (ideal)
            metrics["unsafe_success"] = float(((reached) & (collided)).mean() * 100.0)    # reached BUT collided en route
            metrics["safe_fail"]      = float(((~reached) & (~collided)).mean() * 100.0)  # no collision, didn't reach
            metrics["unsafe_fail"]    = float(((~reached) & (collided)).mean() * 100.0)   # collided and didn't reach
            print(f"2x2 [reached x collided] safe_success={metrics['safe_success']:.1f} "
                  f"unsafe_success={metrics['unsafe_success']:.1f} "
                  f"safe_fail={metrics['safe_fail']:.1f} unsafe_fail={metrics['unsafe_fail']:.1f} (n={int(n)})")

    print(metrics)

    results_path = results_path / cfg.output.filename
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with results_path.open("a") as f:
        f.write("\n")  # separate from previous runs

        f.write("==== CONFIG ====\n")
        f.write(OmegaConf.to_yaml(cfg))
        f.write("\n")

        f.write("==== RESULTS ====\n")
        f.write(f"metrics: {metrics}\n")
        f.write(f"evaluation_time: {end_time - start_time} seconds\n")


if __name__ == "__main__":
    run()
