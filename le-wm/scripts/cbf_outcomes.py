"""2x2 outcomes for CBF trajectory files (collect_trajs.py .h5), paired by trajectory index across files.
Per file: collision (ever inside an obstacle), success, the safe/unsafe x success/fail split, the median final distance
to the goal of the safe failures, and a paired exact McNemar test on collision vs the first file.
Usage: cbf_outcomes.py <file.h5> [...]   (first file = reference)"""
import sys
import numpy as np, h5py
from scipy.stats import binomtest


def load(p):
    with h5py.File(p, "r") as f:
        t = f["trajectories"]
        rows = []
        for k in sorted(t.keys()):
            g = t[k]
            coll = bool(np.asarray(g.attrs["collision"]).any())
            rows.append((coll, bool(g.attrs["success"]), float(g.attrs["final_distance_to_goal"])))
    return rows


files = sys.argv[1:]
data = [load(p) for p in files]
ref = np.array([r[0] for r in data[0]])
for p, rows in zip(files, data):
    c = np.array([r[0] for r in rows]); s = np.array([r[1] for r in rows]); d = np.array([r[2] for r in rows])
    n10, n01 = int((c & ~ref).sum()), int((~c & ref).sum())
    pv = binomtest(n10, n10 + n01).pvalue if n10 + n01 else 1.0
    sf = (~c) & (~s)
    print(f"{p.split('/')[-1]}: collision {c.mean():.3f} success {s.mean():.3f} | safe-succ {((~c) & s).mean():.3f} "
          f"safe-fail {sf.mean():.3f} unsafe-succ {(c & s).mean():.3f} unsafe-fail {(c & ~s).mean():.3f} | "
          f"final dist to goal, safe-fails: median {np.median(d[sf]) if sf.any() else float('nan'):.2f} | "
          f"paired coll vs first {n10}/{n01} p={pv:.2g}", flush=True)
