"""Convert the dubins Dreamer-style buffer (``trajectories/<id>/{observations,
actions, states, failures, rewards}``) into the swm flat dataset layout used by
LE-WM training (``pixels, action, state, proprio, episode_idx, step_idx,
ep_offset, ep_len``).

The output drops into ``$STABLEWM_HOME/datasets/<name>`` and is referenced from
a sibling data config (``le-wm/config/train/data/dubins.yaml``).

dubins schema details:
- ``observations``: (T, H, W, 3) uint8
- ``actions``: (T,) or (T, 1) float — single steering rate
- ``states``: (T, 3) float [x, y, theta]
- ``failures``: (T,) bool / uint8 — 1 when colliding

We expose:
- ``state``: copy of ``states`` (3D)
- ``proprio``: [cos(theta), sin(theta)] — matches what fill_offline_dataset uses
  for ``obs_state`` and gives LE-WM a normalized heading signal.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--src",
        default="/home/seongbin/latent/latent_cbf/data/buffers/dreamer_buffer.h5",
        help="Dreamer-style buffer with `trajectories/<id>/{...}`.",
    )
    parser.add_argument("--out", required=True,
                        help="Path to flat output h5 (place under $STABLEWM_HOME/datasets/).")
    parser.add_argument("--max_trajs", type=int, default=None)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pixels_chunks, action_chunks, state_chunks, proprio_chunks = [], [], [], []
    failures_chunks = []
    ep_offset, ep_len, episode_idx, step_idx = [], [], [], []
    cur = 0

    with h5py.File(args.src, "r") as src:
        trajs = src["trajectories"]
        keys = list(trajs.keys())
        if args.max_trajs is not None:
            keys = keys[: args.max_trajs]
        for i, k in enumerate(tqdm(keys, desc="dubins->swm", ncols=0)):
            tr = trajs[k]
            pixels = tr["observations"][:]
            actions = np.asarray(tr["actions"][:], dtype=np.float32)
            if actions.ndim == 1:
                actions = actions[:, None]
            states = np.asarray(tr["states"][:], dtype=np.float32)
            failures = np.asarray(tr["failures"][:], dtype=np.uint8)
            # Dubins demos store T+1 observations / T actions; trim to the
            # action length so every per-frame field has the same length.
            T = min(states.shape[0], actions.shape[0], pixels.shape[0],
                    failures.shape[0])
            pixels = pixels[:T]
            actions = actions[:T]
            states = states[:T]
            failures = failures[:T]

            theta = states[:, -1]
            proprio = np.stack([np.cos(theta), np.sin(theta)], axis=-1).astype(np.float32)

            pixels_chunks.append(pixels.astype(np.uint8))
            action_chunks.append(actions)
            state_chunks.append(states)
            proprio_chunks.append(proprio)
            failures_chunks.append(failures)
            ep_offset.append(cur)
            ep_len.append(T)
            episode_idx.append(np.full(T, i, dtype=np.int64))
            step_idx.append(np.arange(T, dtype=np.int64))
            cur += T

    pixels_arr = np.concatenate(pixels_chunks, axis=0)
    action_arr = np.concatenate(action_chunks, axis=0)
    state_arr = np.concatenate(state_chunks, axis=0)
    proprio_arr = np.concatenate(proprio_chunks, axis=0)
    failures_arr = np.concatenate(failures_chunks, axis=0)
    episode_idx_arr = np.concatenate(episode_idx, axis=0)
    step_idx_arr = np.concatenate(step_idx, axis=0)
    ep_offset = np.asarray(ep_offset, dtype=np.int64)
    ep_len = np.asarray(ep_len, dtype=np.int64)

    with h5py.File(out_path, "w") as dst:
        dst.create_dataset("pixels", data=pixels_arr)
        dst.create_dataset("action", data=action_arr.astype(np.float32))
        dst.create_dataset("state", data=state_arr.astype(np.float32))
        dst.create_dataset("proprio", data=proprio_arr.astype(np.float32))
        dst.create_dataset("failures", data=failures_arr.astype(np.uint8))
        dst.create_dataset("episode_idx", data=episode_idx_arr)
        dst.create_dataset("step_idx", data=step_idx_arr)
        dst.create_dataset("ep_offset", data=ep_offset)
        dst.create_dataset("ep_len", data=ep_len)

    print(
        f"wrote {out_path}: pixels {pixels_arr.shape}, action {action_arr.shape}, "
        f"state {state_arr.shape}, {len(ep_len)} episodes, {cur} frames"
    )


if __name__ == "__main__":
    main()
