"""Rotation OOD via a REAL 90deg camera rotation (not np.rot90 of the perspective image).
Rotate the fixedfar camera 90 about world-z (same intrinsics/framing, rotated viewpoint) and
render paired base + rotated per state. base matches the training view; rotated is the OOD.
Saves (base, camrot, state, failure)."""
import os, sys, argparse
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, safety_gymnasium as sg, mujoco
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from collect_dataset import full_goal, full_loiter, margin

def qmul(a, b):
    w1,x1,y1,z1=a; w2,x2,y2,z2=b
    return np.array([w1*w2-x1*x2-y1*y2-z1*z2, w1*x2+x1*w2+y1*z2-z1*y2,
                     w1*y2-x1*z2+y1*w2+z1*x2, w1*z2+x1*y2-y1*x2+z1*w2])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True); ap.add_argument("--n", type=int, default=8000)
    ap.add_argument("--size", type=int, default=128); ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--p_loiter", type=float, default=0.4); ap.add_argument("--seed", type=int, default=200)
    a = ap.parse_args()
    env = sg.make("SafetyCarGoal1-v0", render_mode="rgb_array", camera_name="fixedfar", width=a.size, height=a.size)
    u = env.unwrapped; t = u.task; rng = np.random.default_rng(a.seed)
    th = np.pi/2; Rz = np.array([[0,-1,0],[1,0,0],[0,0,1]], float); qz = np.array([np.cos(th/2),0,0,np.sin(th/2)])
    def paired():
        model, data = t.model, t.data
        cid = [i for i in range(model.ncam) if model.camera(i).name=="fixedfar"][0]
        base = u.render().copy()
        op, oq = model.cam_pos[cid].copy(), model.cam_quat[cid].copy()
        model.cam_pos[cid] = Rz@op; model.cam_quat[cid] = qmul(qz, oq); mujoco.mj_forward(model, data)
        rot = u.render().copy()
        model.cam_pos[cid] = op; model.cam_quat[cid] = oq; mujoco.mj_forward(model, data)
        return base.astype(np.uint8), rot.astype(np.uint8)
    US_B, US_R, US_S, SF_B, SF_R, SF_S = [], [], [], [], [], []
    n_us = int(a.n*a.target_unsafe); n_sf = a.n - n_us
    while len(US_B) < n_us or len(SF_B) < n_sf:
        env.reset(seed=int(rng.integers(1<<30))); loit = rng.random() < a.p_loiter
        for _ in range(100):
            m = np.asarray(t.agent.mat).reshape(3,3)
            st = np.array([t.agent.pos[0], t.agent.pos[1], np.arctan2(m[1,0], m[0,0])], np.float32)
            fail = int(margin(t) < 0)
            if fail and len(US_B) < n_us:
                b, r = paired(); US_B.append(b); US_R.append(r); US_S.append(st)
            elif (not fail) and len(SF_B) < n_sf and rng.random() < 0.15:
                b, r = paired(); SF_B.append(b); SF_R.append(r); SF_S.append(st)
            env.step(full_loiter(t, rng) if loit else full_goal(t, rng))
        if (len(US_B)+len(SF_B)) % 1000 < 100: print(f"  us={len(US_B)}/{n_us} sf={len(SF_B)}/{n_sf}", flush=True)
    env.close()
    base = np.array(US_B+SF_B, np.uint8); rot = np.array(US_R+SF_R, np.uint8)
    st = np.array(US_S+SF_S, np.float32); fail = np.array([1]*len(US_B)+[0]*len(SF_B), np.uint8)
    p = rng.permutation(len(base))
    np.savez_compressed(a.out, base=base[p], camrot=rot[p], state=st[p], failure=fail[p])
    print(f"WROTE {a.out} n={len(base)} unsafe={fail.mean():.3f}")

if __name__ == "__main__":
    main()
