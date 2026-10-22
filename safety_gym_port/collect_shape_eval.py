"""Collect a paired-by-seed eval set for the SHAPE (diamond) OOD: hazards as CIRCLE vs BOX.
Hazards are contype=0 (visual-only), so a box-patched env with an identical seed+action stream
produces IDENTICAL agent poses -> the circle and box runs pair by index. Run twice (--shape
circle, --shape box, SAME --seed); the two npz align. blue image + state + failure."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import make_env, render, act, margin

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
    ap.add_argument("--agent", default="Car", choices=["Car", "Point"]); ap.add_argument("--clean", action="store_true")
    ap.add_argument("--ep_steps", type=int, default=100)
    a = ap.parse_args()
    if a.shape == "box": patch_box()
    env = make_env(a.agent, a.size, a.clean)
    u = env.unwrapped; t = u.task; rng = np.random.default_rng(a.seed)
    US_I, US_S, SF_I, SF_S = [], [], [], []
    n_us = int(a.n*a.target_unsafe); n_sf = a.n - n_us
    while len(US_I) < n_us or len(SF_I) < n_sf:
        env.reset(seed=int(rng.integers(1<<30))); loit = rng.random() < a.p_loiter; brake_at = rng.uniform(0, 0.4)
        for _ in range(a.ep_steps):
            m = np.asarray(t.agent.mat).reshape(3,3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1,0], m[0,0])], np.float32)
            fail = int(margin(t) < 0)
            if fail and len(US_I) < n_us:
                US_I.append(render(u, a.clean)); US_S.append(st)
            elif (not fail) and len(SF_I) < n_sf and rng.random() < 0.15:
                SF_I.append(render(u, a.clean)); SF_S.append(st)
            env.step(act(t, rng, a.agent, loit, brake_at))
        if (len(US_I)+len(SF_I)) % 1000 < 100: print(f"  us={len(US_I)}/{n_us} sf={len(SF_I)}/{n_sf}", flush=True)
    env.close()
    img = np.array(US_I + SF_I, np.uint8); st = np.array(US_S + SF_S, np.float32)
    fail = np.array([1]*len(US_I) + [0]*len(SF_I), np.uint8)
    p = rng.permutation(len(img))
    np.savez_compressed(a.out, img=img[p], state=st[p], failure=fail[p], shape=a.shape)
    print(f"WROTE {a.out} shape={a.shape} n={len(img)} unsafe={fail.mean():.3f}")

if __name__ == "__main__":
    main()
