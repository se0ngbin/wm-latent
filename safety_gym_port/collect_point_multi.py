"""Layout-controlled multi-variant eval set for the CLEAN Point study (same protocol as collect_multicolor.py /
sg_plain_eval.py: every variant is a render of the SAME pose, so a train/test split by frame isolates appearance
from the layout-dependent safety signal). All renders are clean (no cost-indicator sphere, no lidar ring).
Per kept Point state:
  base   default hazard colour (= training data)       color  held-out purple hazards
  rotate real 90deg rotation of the fixedfar camera     shape  box hazards (second env, same seed, pose copied)
  agent  Car robot at the same pose (third env, same seed, pose copied)
Saves base/color/rotate/shape/agent (N,H,W,3), state (N,3), velocity (N,2), failure (N,)."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_env import PALETTE
from collect_dataset import make_env, render, act, margin
from collect_multicolor import qmul
from collect_agent_eval import place, heading
import safety_gymnasium.assets.geoms.hazards as hz

_cfg = hz.Hazards.get_config
def _maybe_box(self, xy_pos, rot):   # box hazards only for envs flagged .box (patch is class-global, layouts are rebuilt per reset)
    c = _cfg(self, xy_pos, rot)
    if getattr(self, "box", False): c["geoms"][0]["type"] = "box"; c["geoms"][0]["size"] = [float(self.size), float(self.size), 1e-2]
    return c
hz.Hazards.get_config = _maybe_box

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.6); ap.add_argument("--ep_steps", type=int, default=200)
    ap.add_argument("--seed", type=int, default=321)
    a = ap.parse_args()
    env, benv, cenv = make_env("Point", a.size, True), make_env("Point", a.size, True), make_env("Car", a.size, True)
    benv.unwrapped.task.hazards.box = True
    u, t = env.unwrapped, env.unwrapped.task
    purp = np.array(PALETTE["purple"], float)
    th = np.pi / 2; Rz = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]], float); qz = np.array([np.cos(th / 2), 0, 0, np.sin(th / 2)])
    rng = np.random.default_rng(a.seed)

    def copy_scene(ou):   # goal + vase + agent pose of the base env into another env
        ot = ou.task; m, d = ot.model, ot.data
        m.body("goal").pos[:2] = t.goal.pos[:2]
        ja = m.joint(m.body("vase0").jntadr[0]).qposadr[0]
        d.qpos[ja:ja + 3] = t.data.body("vase0").xpos; d.qpos[ja + 3:ja + 7] = t.data.body("vase0").xquat
        place(ot, t.agent.pos[:2].copy(), heading(t))
        return render(ou, True)

    def snapshot():
        model, data = t.model, t.data
        haz = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("hazard")]
        base = render(u, True)
        orig = model.geom_rgba[haz].copy(); model.geom_rgba[haz] = purp
        color = render(u, True); model.geom_rgba[haz] = orig
        cid = [i for i in range(model.ncam) if model.camera(i).name == "fixedfar"][0]
        op, oq = model.cam_pos[cid].copy(), model.cam_quat[cid].copy()
        model.cam_pos[cid] = Rz @ op; model.cam_quat[cid] = qmul(qz, oq); mujoco.mj_forward(model, data)
        rot = render(u, True)
        model.cam_pos[cid] = op; model.cam_quat[cid] = oq; mujoco.mj_forward(model, data)
        return base, color, rot, copy_scene(benv.unwrapped), copy_scene(cenv.unwrapped)

    US, SF = [], []
    n_us = int(a.n * a.target_unsafe); n_sf = a.n - n_us; skipped = 0
    while len(US) < n_us or len(SF) < n_sf:
        s = int(rng.integers(1 << 30)); env.reset(seed=s); benv.reset(seed=s); cenv.reset(seed=s)
        if not all(np.allclose(np.asarray(t.hazards.pos), np.asarray(e.unwrapped.task.hazards.pos)) for e in (benv, cenv)):
            skipped += 1; continue
        loit = rng.random() < a.p_loiter; brake_at = rng.uniform(0, 0.4)
        for _ in range(a.ep_steps):
            fail = int(margin(t) < 0)
            if (fail and len(US) < n_us) or ((not fail) and len(SF) < n_sf and rng.random() < 0.15):
                st = np.array([t.agent.pos[0], t.agent.pos[1], heading(t)], np.float32)
                (US if fail else SF).append(snapshot() + (st, np.asarray(t.agent.vel[:2], np.float32)))
            env.step(act(t, rng, "Point", loit, brake_at))
        if (len(US) + len(SF)) % 500 < 50: print(f"  us={len(US)}/{n_us} sf={len(SF)}/{n_sf} skipped_layouts={skipped}", flush=True)
    for e in (env, benv, cenv): e.close()
    rows = US + SF; p = rng.permutation(len(rows))   # shuffled -> the downstream 80/20 frame split mixes layouts like collect_multicolor
    keys = ["base", "color", "rotate", "shape", "agent", "state", "velocity"]
    out = {k: np.array([rows[i][j] for i in p]) for j, k in enumerate(keys)}
    out["failure"] = np.array([1] * len(US) + [0] * len(SF), np.uint8)[p]
    np.savez_compressed(a.out, **out)
    print(f"WROTE {a.out} n={len(p)} unsafe={out['failure'].mean():.3f} skipped_layouts={skipped}")

if __name__ == "__main__":
    main()
