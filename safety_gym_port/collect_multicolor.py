"""Multi-color sg set for the robust-margin study (Ilyas-style), with LAYOUT CONTROLLED: every
appearance variant is a render of the SAME pose, so train/test split by frame isolates appearance
from the (random, layout-dependent) safety signal. Per kept state, render the scene under every
training color (blue/orange/teal/yellow/brown/green), under the held-out OOD color (purple), and
under a REAL 90deg camera rotation (blue hazards). Saves: imgs (N,K,H,W,3) train colors, purple
(N,H,W,3), camrot (N,H,W,3), colors (K,), state (N,3), failure (N,).
Downstream splits frames -> train on train-frames (blue/rand/worst colors), test on test-frames
purple (same-type OOD) + camrot (cross-type OOD). base view for the rotation test = imgs[:,blue]."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sg_env import PALETTE
from collect_dataset import full_goal, full_loiter, margin

TRAIN_COLORS = ["blue", "orange", "teal", "yellow", "brown", "green"]  # purple held out

def qmul(a, b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=5000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.4); ap.add_argument("--seed", type=int, default=321)
    a = ap.parse_args()
    env = sg.make("SafetyCarGoal1-v0", render_mode="rgb_array", camera_name="fixedfar", width=a.size, height=a.size)
    u = env.unwrapped; t = u.task
    cols = {c: np.array(PALETTE[c], float) for c in TRAIN_COLORS}
    purp = np.array(PALETTE["purple"], float); blue = cols["blue"]
    th = np.pi/2; Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]], float); qz = np.array([np.cos(th/2),0,0,np.sin(th/2)])
    rng = np.random.default_rng(a.seed)
    US, SF = [], []   # each entry: (train_stack(K,H,W,3), purple(H,W,3), camrot(H,W,3), state)
    n_us = int(a.n*a.target_unsafe); n_sf = a.n - n_us
    def snapshot():
        model, data = t.model, t.data
        haz = [i for i in range(model.ngeom) if (model.geom(i).name or "").startswith("hazard")]
        frames = []
        for c in TRAIN_COLORS:
            for i in haz: model.geom_rgba[i] = cols[c]
            frames.append(u.render().copy().astype(np.uint8))
        for i in haz: model.geom_rgba[i] = purp
        purple_img = u.render().copy().astype(np.uint8)
        for i in haz: model.geom_rgba[i] = blue
        # real 90deg camera rotation (blue hazards)
        cid = [i for i in range(model.ncam) if model.camera(i).name=="fixedfar"][0]
        op, oq = model.cam_pos[cid].copy(), model.cam_quat[cid].copy()
        model.cam_pos[cid] = Rz@op; model.cam_quat[cid] = qmul(qz, oq); mujoco.mj_forward(model, data)
        camrot_img = u.render().copy().astype(np.uint8)
        model.cam_pos[cid] = op; model.cam_quat[cid] = oq; mujoco.mj_forward(model, data)
        return np.stack(frames, 0), purple_img, camrot_img
    while len(US) < n_us or len(SF) < n_sf:
        env.reset(seed=int(rng.integers(1 << 30))); loit = rng.random() < a.p_loiter
        for _ in range(100):
            m = np.asarray(t.agent.mat).reshape(3, 3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1, 0], m[0, 0])], np.float32)
            fail = int(margin(t) < 0)
            if fail and len(US) < n_us:
                US.append((*snapshot(), st))
            elif (not fail) and len(SF) < n_sf and rng.random() < 0.15:
                SF.append((*snapshot(), st))
            env.step((full_loiter(t, rng) if loit else full_goal(t, rng)))
        if (len(US)+len(SF)) % 1000 < 100: print(f"  us={len(US)}/{n_us} sf={len(SF)}/{n_sf}", flush=True)
    env.close()
    rows = US + SF
    imgs = np.stack([r[0] for r in rows]).astype(np.uint8)
    purple = np.stack([r[1] for r in rows]).astype(np.uint8)
    camrot = np.stack([r[2] for r in rows]).astype(np.uint8)
    st_a = np.stack([r[3] for r in rows]).astype(np.float32)
    fail_a = np.array([1]*len(US) + [0]*len(SF), np.uint8)
    p = rng.permutation(len(rows))
    np.savez_compressed(a.out, imgs=imgs[p], purple=purple[p], camrot=camrot[p],
                        colors=np.array(TRAIN_COLORS), state=st_a[p], failure=fail_a[p])
    print(f"WROTE {a.out} n={len(rows)} K={imgs.shape[1]} unsafe={fail_a.mean():.3f} imgs={imgs.shape}")

if __name__ == "__main__":
    main()
