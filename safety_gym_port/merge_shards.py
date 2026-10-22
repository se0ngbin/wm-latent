"""Merge shard h5s into one dubins-schema dataset, renumbering episodes/offsets."""
import sys, glob, h5py, numpy as np
shards = sorted(glob.glob(sys.argv[1])); out = sys.argv[2]
keys = ["pixels","action","proprio","state","failures","step_idx"]
with h5py.File(shards[0], "r") as f0:
    if "velocity" in f0: keys.append("velocity")   # Point datasets (collect_dataset --agent Point)
acc = {k: [] for k in keys}; EPI=[]; ep_len=[]; ep_off=[]; off=0; ep=0
for s in shards:
    with h5py.File(s,"r") as f:
        for k in keys: acc[k].append(f[k][:])
        el = f["ep_len"][:]
        for L in el:
            EPI.append(np.full(int(L), ep, np.int64)); ep_len.append(int(L)); ep_off.append(off)
            off += int(L); ep += 1
    print("merged", s, "episodes now", ep, flush=True)
with h5py.File(out,"w") as f:
    for k in keys:
        data = np.concatenate(acc[k])
        f.create_dataset(k, data=data
                         )
    f.create_dataset("episode_idx", data=np.concatenate(EPI))
    f.create_dataset("ep_len", data=np.array(ep_len,np.int64))
    f.create_dataset("ep_offset", data=np.array(ep_off,np.int64))
    print("WROTE", out, "frames", off, "episodes", ep, "unsafe", round(float(f["failures"][:].mean()),3))
