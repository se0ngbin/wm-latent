"""Agent-shift OOD: the same scene rendered with the Car (training agent) vs the Point robot.
Car drives (same goal/loiter mix as the other eval sets); at each sampled state a Point env reset
with the SAME seed (hazard layout checked equal) gets the Car's goal, vase pose and agent (x,y,theta)
copied in, so only the robot differs. Labels = position margin (identical for both).
base/point are rendered WITHOUT the cost-indicator sphere; base_ind is the Car frame WITH it (what the
existing sg datasets contain). Saves (base, point, state, failure, base_ind)."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import full_goal, full_loiter, margin

def heading(t):
    m = np.asarray(t.agent.mat).reshape(3, 3); return np.arctan2(m[1, 0], m[0, 0])

def wrap(x): return (x + np.pi) % (2 * np.pi) - np.pi

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.4); ap.add_argument("--seed", type=int, default=300)
    a = ap.parse_args()
    mk = lambda ag: sg.make(f"Safety{ag}Goal1-v0", render_mode="rgb_array", camera_name="fixedfar", width=a.size, height=a.size)
    env, penv = mk("Car"), mk("Point")
    u, pu = env.unwrapped, penv.unwrapped; t, pt = u.task, pu.task; rng = np.random.default_rng(a.seed)

    def point_frame():
        m, d = pt.model, pt.data
        m.body("goal").pos[:2] = t.goal.pos[:2]
        ja = m.joint(m.body("vase0").jntadr[0]).qposadr[0]
        d.qpos[ja:ja + 3] = t.data.body("vase0").xpos; d.qpos[ja + 3:ja + 7] = t.data.body("vase0").xquat
        jx, jy, jz = (m.joint(n) for n in ("x", "y", "z"))
        mujoco.mj_forward(m, d)
        A = np.stack([d.xaxis[jx.id][:2], d.xaxis[jy.id][:2]], 1)   # Point slide axes are rotated at reset
        dq = np.linalg.solve(A, t.agent.pos[:2] - pt.agent.pos[:2])
        d.qpos[jx.qposadr[0]] += dq[0]; d.qpos[jy.qposadr[0]] += dq[1]
        d.qpos[jz.qposadr[0]] += wrap(heading(t) - heading(pt)); d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        assert np.allclose(pt.agent.pos[:2], t.agent.pos[:2], atol=1e-5) and abs(wrap(heading(pt) - heading(t))) < 1e-4
        return clean(pu)

    def clean(env_u):   # render WITHOUT safety-gym's red cost-indicator sphere (drawn when last-step cost>0 -> label leak)
        c, env_u.cost = env_u.cost, {}
        img = env_u.render().astype(np.uint8); env_u.cost = c
        return img

    US, SF = [], []
    n_us = int(a.n * a.target_unsafe); n_sf = a.n - n_us; skipped = 0
    while len(US) < n_us or len(SF) < n_sf:
        s = int(rng.integers(1 << 30)); env.reset(seed=s); penv.reset(seed=s)
        if not np.allclose(np.asarray(t.hazards.pos), np.asarray(pt.hazards.pos)): skipped += 1; continue
        loit = rng.random() < a.p_loiter
        for _ in range(100):
            st = np.array([t.agent.pos[0], t.agent.pos[1], heading(t)], np.float32); fail = int(margin(t) < 0)
            if (fail and len(US) < n_us) or ((not fail) and len(SF) < n_sf and rng.random() < 0.15):
                (US if fail else SF).append((clean(u), point_frame(), st, u.render().astype(np.uint8)))
            env.step(full_loiter(t, rng) if loit else full_goal(t, rng))
        if (len(US) + len(SF)) % 1000 < 100: print(f"  us={len(US)}/{n_us} sf={len(SF)}/{n_sf} skipped_layouts={skipped}", flush=True)
    env.close(); penv.close()
    rows = US + SF; p = rng.permutation(len(rows))
    base, point, st, base_ind = (np.array([rows[i][k] for i in p]) for k in range(4))
    fail = np.array([1] * len(US) + [0] * len(SF), np.uint8)[p]
    np.savez_compressed(a.out, base=base, point=point, state=st, failure=fail, base_ind=base_ind)
    print(f"WROTE {a.out} n={len(base)} unsafe={fail.mean():.3f} skipped_layouts={skipped}")

if __name__ == "__main__":
    main()
