"""Point env server for CEM planning (runs in the safety-gym venv; le-wm/scripts/sg_point_cem.py drives it over stdin/stdout pickle).
One env per episode, overhead@224 clean renders (the WM training camera). Episode from a seed: env layout + start from reset(seed);
goal = a point on the far side of a random hazard, so the straight start->goal line goes through that hazard's centre.
Commands (pickled tuples): ("reset", seeds) -> [(img, goal_img, info)]; ("step", actions (N,2)) -> [(img, margin, pos)]."""
import os, sys, pickle
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import make_env, render, margin
from collect_agent_eval import place, heading

ARENA = 1.4   # goals kept inside |x|,|y| < ARENA (overhead frame spans 1.8)


GOAL_MODE = os.environ.get("GOAL_MODE", "hazard")                       # hazard: behind a hazard; ahead: straight ahead of the start (sanity)
GOAL_D = [float(x) for x in os.environ.get("GOAL_D", "0.5,1.1").split(",")]   # allowed |goal - start|


def pick_goal(t, rng, tries=2000):
    """hazard: g = c + u * d, u = unit(c - start), d ~ U(r+.2, r+.45) (straight line crosses hazard c); ahead: g = start + heading * U(GOAL_D).
    Both: |g - start| in GOAL_D, g inside the arena and clear of all hazards by .15."""
    s = np.asarray(t.agent.pos[:2]); hs = np.asarray([h[:2] for h in t.hazards.pos]); r = float(t.hazards.size)
    for _ in range(tries):
        if GOAL_MODE == "ahead":
            u = np.array([np.cos(heading(t)), np.sin(heading(t))]); g = s + u * rng.uniform(*GOAL_D)
        else:
            c = hs[rng.integers(len(hs))]; u = (c - s) / np.linalg.norm(c - s); g = c + u * rng.uniform(r + .2, r + .45)
        if GOAL_D[0] <= np.linalg.norm(g - s) <= GOAL_D[1] and np.abs(g).max() < ARENA and (np.linalg.norm(hs - g, axis=1) - r).min() > .15:
            return g, np.arctan2(u[1], u[0])
    return None


def episode(seed, env):
    """Re-seeds pooled env slot `env` (never closed: closing the first-created env blacks out every later render under EGL).
    Returns (start img, goal img, info) or None if this layout has no valid goal / start is unsafe."""
    env.reset(seed=int(seed)); u = env.unwrapped; t = u.task
    gp = pick_goal(t, np.random.default_rng(seed)) if margin(t) >= .1 else None
    if gp is None: return None
    s, th = np.asarray(t.agent.pos[:2]).copy(), heading(t)
    place(t, gp[0], gp[1]); gimg = render(u, True)          # goal image: agent at the goal, facing the travel direction
    place(t, s, th); img = render(u, True)                  # back to the start, at rest
    return img, gimg, dict(seed=int(seed), start=s, goal=gp[0], goal_th=gp[1], hazards=np.asarray([h[:2] for h in t.hazards.pos]), r=float(t.hazards.size))


def main():
    inp, out = sys.stdin.buffer, sys.stdout.buffer
    sys.stdout = sys.stderr   # stray prints must not corrupt the pickle stream
    pool, envs = [], []
    while True:
        try: cmd, arg = pickle.load(inp)
        except EOFError: return
        if cmd == "reset":
            envs, res = [], []
            for s in arg:   # valid episodes fill pool slots in order; an invalid seed leaves its slot for the next seed
                if len(pool) <= len(envs): pool.append(make_env("Point", 224, True, "overhead"))
                e = episode(s, pool[len(envs)])
                if e is not None: envs.append(pool[len(envs)])
                res.append(e)
        elif cmd == "step":
            res = []
            for e, a in zip(envs, arg):
                e.step(np.clip(np.asarray(a, dtype=np.float64), -1, 1)); t = e.unwrapped.task
                res.append((render(e.unwrapped, True), margin(t), np.asarray(t.agent.pos[:2]).copy()))
        pickle.dump(res, out); out.flush()


if __name__ == "__main__":
    main()
