"""Collect a SafetyCarGoal1 offline dataset in the dubins_expert.h5 schema, matched to the
dubins distribution: top-down pixels, goal-reaching + hazard-loiter trajectories, and
CHUNK-LEVEL FILTERING to hit ~17% unsafe frames (dubins' rate) on the default geometry.
Split each episode into non-overlapping W-frame chunks (short trajectories the WM trains on);
keep every chunk containing an unsafe frame, subsample all-safe chunks to the target ratio.
Runs in the safety-gym venv; le-wm trains offline on the h5."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, h5py
import safety_gymnasium as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy import goal_seek_action, loiter_action, _wrap


def full_goal(t, rng):
    return goal_seek_action(t, rng, forward=1.0, noise=0.3)


def full_loiter(t, rng):
    x, y = float(t.agent.pos[0]), float(t.agent.pos[1])
    m = np.asarray(t.agent.mat).reshape(3, 3); th = np.arctan2(m[1, 0], m[0, 0])
    hs = np.asarray([h[:2] for h in t.hazards.pos]); j = np.linalg.norm(hs - [x, y], axis=1).argmin()
    e = _wrap(np.arctan2(hs[j][1] - y, hs[j][0] - x) - th); k = np.clip(2.0 * e, -1, 1)
    return np.clip(np.array([1.0 + k, 1.0 - k]) + rng.normal(0, 0.3, 2), -1, 1)


def margin(t):
    r = float(t.hazards.size)
    return min(np.linalg.norm(np.asarray(t.agent.pos[:2]) - np.asarray(h[:2])) - r for h in t.hazards.pos)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=170000, help="target FINAL frames after filtering")
    ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--ep_steps", type=int, default=100)
    ap.add_argument("--chunk", type=int, default=8)
    ap.add_argument("--p_loiter", type=float, default=0.4)
    ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--color", default="blue")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = np.random.default_rng(a.seed)
    env = sg.make("SafetyCarGoal1-v0", render_mode="rgb_array", camera_name="fixedfar",
                  width=a.size, height=a.size)
    t = env.unwrapped.task
    import safety_gymnasium.assets.color as C
    from sg_env import PALETTE
    col = np.array(PALETTE[a.color], float)

    # pools of non-overlapping chunks, each = (pix[W], act[W], state[W], fail[W])
    unsafe_chunks, safe_chunks = [], []
    n_unsafe_f = n_safe_in_unsafe = 0
    ep = 0
    def enough():   # target reached?
        keep_safe = int(round(n_unsafe_f * (1 - a.target_unsafe) / a.target_unsafe)) - n_safe_in_unsafe
        have = n_unsafe_f + n_safe_in_unsafe + min(len(safe_chunks) * a.chunk, max(keep_safe, 0))
        return have >= a.frames and n_unsafe_f > 0

    while not enough():
        t.hazards.color = col; C.COLOR["hazard"] = col
        env.reset(seed=int(rng.integers(1 << 30)))
        t.hazards.color = col
        loit = rng.random() < a.p_loiter
        P, A, S, F = [], [], [], []
        for _ in range(a.ep_steps):
            st = env.state() if hasattr(env, "state") else None
            m = np.asarray(t.agent.mat).reshape(3, 3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1, 0], m[0, 0])], np.float32)
            img = env.unwrapped.render().astype(np.uint8)
            act = (full_loiter(t, rng) if loit else full_goal(t, rng)).astype(np.float32)
            P.append(img); A.append(act); S.append(st); F.append(int(margin(t) < 0))
            env.step(act)
        # split into non-overlapping chunks
        W = a.chunk
        for i in range(0, len(P) - W + 1, W):
            f = np.array(F[i:i + W])
            chunk = (np.array(P[i:i + W], np.uint8), np.array(A[i:i + W], np.float32),
                     np.array(S[i:i + W], np.float32), f.astype(np.uint8))
            if f.any():
                unsafe_chunks.append(chunk); n_unsafe_f += int(f.sum()); n_safe_in_unsafe += int((1 - f).sum())
            else:
                safe_chunks.append(chunk)
        ep += 1
        if ep % 50 == 0:
            print(f"  ep={ep} unsafe_f={n_unsafe_f} unsafe_chunks={len(unsafe_chunks)} safe_chunks={len(safe_chunks)}", flush=True)
    env.close()

    # subsample safe-only chunks to hit target unsafe frame fraction
    keep_safe_frames = int(round(n_unsafe_f * (1 - a.target_unsafe) / a.target_unsafe)) - n_safe_in_unsafe
    n_safe_chunks_keep = max(0, keep_safe_frames // a.chunk)
    rng.shuffle(safe_chunks)
    kept = unsafe_chunks + safe_chunks[:n_safe_chunks_keep]
    rng.shuffle(kept)

    PIX = np.concatenate([c[0] for c in kept]); ACT = np.concatenate([c[1] for c in kept])
    STA = np.concatenate([c[2] for c in kept]); FAIL = np.concatenate([c[3] for c in kept])
    PRO = np.stack([np.cos(STA[:, 2]), np.sin(STA[:, 2])], 1).astype(np.float32)
    W = a.chunk; nchunk = len(kept)
    EPI = np.repeat(np.arange(nchunk), W); STP = np.tile(np.arange(W), nchunk)
    ep_len = np.full(nchunk, W, np.int64); ep_off = np.arange(nchunk) * W

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        f.create_dataset("pixels", data=PIX, )
        f.create_dataset("action", data=ACT); f.create_dataset("proprio", data=PRO)
        f.create_dataset("state", data=STA); f.create_dataset("failures", data=FAIL)
        f.create_dataset("episode_idx", data=EPI); f.create_dataset("step_idx", data=STP)
        f.create_dataset("ep_len", data=ep_len); f.create_dataset("ep_offset", data=ep_off)
    print(f"WROTE {a.out} frames={len(PIX)} chunks={nchunk} unsafe_frac={FAIL.mean():.3f} "
          f"(from {ep} episodes)")


if __name__ == "__main__":
    main()
