"""Paired OOD eval set for the appearance-shift margin study, matched to the training
distribution. Roll goal/loiter-mix trajectories; keep unsafe + subsample safe to target
unsafe_frac; for each kept state render the SAME scene in-dist (blue) and OOD (held-out)
via LIVE geom recolor (identical pose). Saves npz (blue, ood, state, failure). le-wm venv
then encodes both -> margin zero-shot AUC + cos_la."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_env import PALETTE
from collect_dataset import full_goal, full_loiter, margin

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--ood_color", default="purple")
    ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.4); ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()
    env = sg.make("SafetyCarGoal1-v0", render_mode="rgb_array", camera_name="fixedfar", width=a.size, height=a.size)
    u = env.unwrapped; t = u.task
    ood = np.array(PALETTE[a.ood_color], float); blue = np.array(PALETTE["blue"], float)
    rng = np.random.default_rng(a.seed)
    US_B, US_O, US_S, SF_B, SF_O, SF_S = [], [], [], [], [], []   # unsafe/safe pools
    n_us_keep = int(a.n * a.target_unsafe); n_sf_keep = a.n - n_us_keep
    def paired():
        model = u.task.model; haz = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("hazard")]
        for i in haz: model.geom_rgba[i] = blue
        ib = u.render().copy()
        for i in haz: model.geom_rgba[i] = ood
        io = u.render().copy()
        for i in haz: model.geom_rgba[i] = blue
        return ib.astype(np.uint8), io.astype(np.uint8)
    while len(US_B) < n_us_keep or len(SF_B) < n_sf_keep:
        env.reset(seed=int(rng.integers(1 << 30))); loit = rng.random() < a.p_loiter
        for _ in range(100):
            m = np.asarray(t.agent.mat).reshape(3, 3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1, 0], m[0, 0])], np.float32)
            fail = int(margin(t) < 0)
            if fail and len(US_B) < n_us_keep:
                ib, io = paired(); US_B.append(ib); US_O.append(io); US_S.append(st)
            elif (not fail) and len(SF_B) < n_sf_keep and rng.random() < 0.15:
                ib, io = paired(); SF_B.append(ib); SF_O.append(io); SF_S.append(st)
            env.step((full_loiter(t, rng) if loit else full_goal(t, rng)))
        if (len(US_B) + len(SF_B)) % 1000 < 100:
            print(f"  unsafe={len(US_B)}/{n_us_keep} safe={len(SF_B)}/{n_sf_keep}", flush=True)
    env.close()
    blue_a = np.array(US_B + SF_B, np.uint8); ood_a = np.array(US_O + SF_O, np.uint8)
    st_a = np.array(US_S + SF_S, np.float32); fail_a = np.array([1]*len(US_B) + [0]*len(SF_B), np.uint8)
    p = rng.permutation(len(blue_a))
    np.savez_compressed(a.out, blue=blue_a[p], ood=ood_a[p], state=st_a[p], failure=fail_a[p], ood_color=a.ood_color)
    print(f"WROTE {a.out} n={len(blue_a)} unsafe_frac={fail_a.mean():.3f}")

if __name__ == "__main__":
    main()
