"""Collect a paired-by-seed eval set for the SHAPE (diamond) OOD: hazards as CIRCLE vs BOX.
Hazards are contype=0 (visual-only), so a box-patched env with an identical seed+action stream
produces IDENTICAL agent poses -> the circle and box runs pair by index. Run twice (--shape
circle, --shape box, SAME --seed); the two npz align. blue image + state + failure."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import full_goal, full_loiter, margin

def patch_box():
    import safety_gymnasium.assets.geoms.hazards as hz
    orig = hz.Hazards.get_config
    def box(self, xy_pos, rot):
        c = orig(self, xy_pos, rot); c["geoms"][0]["type"] = "box"
        c["geoms"][0]["size"] = [float(self.size), float(self.size), 1e-2]; return c
    hz.Hazards.get_config = box

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--shape", choices=["circle","box"], required=True)
    ap.add_argument("--n", type=int, default=8000); ap.add_argument("--size", type=int, default=128)
    ap.add_argument("--target_unsafe", type=float, default=0.17); ap.add_argument("--p_loiter", type=float, default=0.4)
    ap.add_argument("--seed", type=int, default=200)
    a = ap.parse_args()
    if a.shape == "box": patch_box()
    env = sg.make("SafetyCarGoal1-v0", render_mode="rgb_array", camera_name="fixedfar", width=a.size, height=a.size)
    u = env.unwrapped; t = u.task; rng = np.random.default_rng(a.seed)
    US_I, US_S, SF_I, SF_S = [], [], [], []
    n_us = int(a.n*a.target_unsafe); n_sf = a.n - n_us
    while len(US_I) < n_us or len(SF_I) < n_sf:
        env.reset(seed=int(rng.integers(1<<30))); loit = rng.random() < a.p_loiter
        for _ in range(100):
            m = np.asarray(t.agent.mat).reshape(3,3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1,0], m[0,0])], np.float32)
            fail = int(margin(t) < 0)
            if fail and len(US_I) < n_us:
                US_I.append(u.render().copy().astype(np.uint8)); US_S.append(st)
            elif (not fail) and len(SF_I) < n_sf and rng.random() < 0.15:
                SF_I.append(u.render().copy().astype(np.uint8)); SF_S.append(st)
            env.step(full_loiter(t, rng) if loit else full_goal(t, rng))
        if (len(US_I)+len(SF_I)) % 1000 < 100: print(f"  us={len(US_I)}/{n_us} sf={len(SF_I)}/{n_sf}", flush=True)
    env.close()
    img = np.array(US_I + SF_I, np.uint8); st = np.array(US_S + SF_S, np.float32)
    fail = np.array([1]*len(US_I) + [0]*len(SF_I), np.uint8)
    p = rng.permutation(len(img))
    np.savez_compressed(a.out, img=img[p], state=st[p], failure=fail[p], shape=a.shape)
    print(f"WROTE {a.out} shape={a.shape} n={len(img)} unsafe={fail.mean():.3f}")

if __name__ == "__main__":
    main()
