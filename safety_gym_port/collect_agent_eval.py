"""Agent-shift OOD: the same scene rendered with the training agent (--base) vs the other robot.
The base agent drives (same goal/loiter mix as the training data); at each sampled state the other
agent's env, reset with the SAME seed (hazard layout checked equal), gets the goal, vase pose and agent
(x,y,theta) copied in, so only the robot differs. Labels = position margin (identical for both).
base/other are rendered WITHOUT the cost-indicator sphere / lidar ring; base_ind is the base frame WITH
the sphere (what the pre-2026-10-22 sg datasets contain). Saves (base, other, state, failure, base_ind)."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import make_env, render, act, margin

def heading(t):
    m = np.asarray(t.agent.mat).reshape(3, 3); return np.arctan2(m[1, 0], m[0, 0])

def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi

def place(t, xy, th):
    """Put the agent of task t at (xy, heading th) with zero velocity."""
    m, d = t.model, t.data
    if t.agent.__class__.__name__ == "Point":   # slide x/y + hinge z; slide axes are rotated at reset
        jx, jy, jz = (m.joint(n) for n in ("x", "y", "z"))
        mujoco.mj_forward(m, d)
        A = np.stack([d.xaxis[jx.id][:2], d.xaxis[jy.id][:2]], 1)
        dq = np.linalg.solve(A, xy - t.agent.pos[:2])
        d.qpos[jx.qposadr[0]] += dq[0]; d.qpos[jy.qposadr[0]] += dq[1]; d.qpos[jz.qposadr[0]] += wrap(th - heading(t))
    else:                                         # Car: free joint 'agent' = [x y z qw qx qy qz]
        ja = m.joint("agent").qposadr[0]
        d.qpos[ja:ja + 2] = xy; d.qpos[ja + 3:ja + 7] = [np.cos(th / 2), 0, 0, np.sin(th / 2)]
    d.qvel[:] = 0; mujoco.mj_forward(m, d)
    assert np.allclose(t.agent.pos[:2], xy, atol=1e-5) and abs(wrap(heading(t) - th)) < 1e-3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--base", default="Car", choices=["Car", "Point"]); ap.add_argument("--ep_steps", type=int, default=100)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.4); ap.add_argument("--seed", type=int, default=300)
    a = ap.parse_args()
    other = "Point" if a.base == "Car" else "Car"
    env, oenv = make_env(a.base, a.size, True), make_env(other, a.size, True)
    u, ou = env.unwrapped, oenv.unwrapped; t, ot = u.task, ou.task; rng = np.random.default_rng(a.seed)

    def other_frame():
        m, d = ot.model, ot.data
        m.body("goal").pos[:2] = t.goal.pos[:2]
        ja = m.joint(m.body("vase0").jntadr[0]).qposadr[0]
        d.qpos[ja:ja + 3] = t.data.body("vase0").xpos; d.qpos[ja + 3:ja + 7] = t.data.body("vase0").xquat
        place(ot, t.agent.pos[:2].copy(), heading(t))
        return render(ou, True)

    US, SF = [], []
    n_us = int(a.n * a.target_unsafe); n_sf = a.n - n_us; skipped = 0
    while len(US) < n_us or len(SF) < n_sf:
        s = int(rng.integers(1 << 30)); env.reset(seed=s); oenv.reset(seed=s)
        if not np.allclose(np.asarray(t.hazards.pos), np.asarray(ot.hazards.pos)): skipped += 1; continue
        loit = rng.random() < a.p_loiter; brake_at = rng.uniform(0, 0.4)
        for _ in range(a.ep_steps):
            st = np.array([t.agent.pos[0], t.agent.pos[1], heading(t)], np.float32); fail = int(margin(t) < 0)
            if (fail and len(US) < n_us) or ((not fail) and len(SF) < n_sf and rng.random() < 0.15):
                (US if fail else SF).append((render(u, True), other_frame(), st, render(u, False)))
            env.step(act(t, rng, a.base, loit, brake_at))
        if (len(US) + len(SF)) % 1000 < 100: print(f"  us={len(US)}/{n_us} sf={len(SF)}/{n_sf} skipped_layouts={skipped}", flush=True)
    env.close(); oenv.close()
    rows = US + SF; p = rng.permutation(len(rows))
    base, oth, st, base_ind = (np.array([rows[i][k] for i in p]) for k in range(4))
    fail = np.array([1] * len(US) + [0] * len(SF), np.uint8)[p]
    np.savez_compressed(a.out, base=base, other=oth, state=st, failure=fail, base_ind=base_ind, base_agent=a.base)
    print(f"WROTE {a.out} base={a.base} n={len(base)} unsafe={fail.mean():.3f} skipped_layouts={skipped}")

if __name__ == "__main__":
    main()
