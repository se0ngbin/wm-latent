"""Convert the flat sg_cargoal_expert.h5 (le-wm schema) to the trajectory-grouped h5 that
dreamer_offline.py's fill_offline_dataset expects: trajectories/traj_i/{observations, states,
actions, failures, rewards}. states = (x,y,theta) (loader reads states[:,-1] as theta).
One trajectory per 8-frame chunk (episode_idx). Same data as LEWM -> valid comparison."""
import sys, h5py, numpy as np
IN = sys.argv[1] if len(sys.argv) > 1 else "/data/seongbin/lewm/datasets/sg_cargoal_expert.h5"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/data/seongbin/lewm/datasets/sg_cargoal_dreamer.h5"
with h5py.File(IN, "r") as f:
    pix = f["pixels"][:]; act = f["action"][:]; st = f["state"][:]; fail = f["failures"][:]
    off = f["ep_offset"][:]; ln = f["ep_len"][:]
n = len(ln)
with h5py.File(OUT, "w") as g:
    tr = g.create_group("trajectories")
    for i in range(n):
        s, L = int(off[i]), int(ln[i]); e = s + L
        t = tr.create_group(f"traj_{i}")
        t.create_dataset("observations", data=pix[s:e], )
        t.create_dataset("states", data=st[s:e].astype(np.float32))
        t.create_dataset("actions", data=act[s:e].astype(np.float32))
        t.create_dataset("failures", data=fail[s:e].astype(np.float32))
        t.create_dataset("rewards", data=np.zeros(L, np.float32))
        if (i + 1) % 5000 == 0: print(f"  {i+1}/{n}", flush=True)
print(f"WROTE {OUT} trajectories={n}")
