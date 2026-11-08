"""Separate "cannot plan under the shift" from "unsafe under the shift" for reacher_cem.py runs (MODE=data, saved .npz).
Goals are rebuilt from reacher_cem.py's deterministic pair selection (held-out episodes, safe endpoints, recorded path crosses
the box, PAIR_SEED 42, 50 pairs). Per run:
  outcome 2x2   : SS = reached goal, never in box; SF = never in box, no goal (cannot plan); US = goal but entered box; UF = box, no goal
  progress      : median fraction of the initial joint-space goal distance (max over joints, wrapped) closed by the closest approach
  path          : median joint-space path length (rad, sum over steps of the max joint step)
  stall         : fraction of SF episodes with progress < 0.25 (no progress rather than a near miss)
  attribution (runs saved with h logging, collision episodes only; entry = first step in the box, plan = the one returned at the
  replan before entry): fooled = that plan's min_k h(zhat_k) > 0 (the model judged it safe); forced = <= 0 (no feasible plan, CCEM
  fell back); noticed = h on the real frame at the entry step < 0 (the margin function sees the box once inside)
Usage: reacher_outcomes.py <npz> [<npz> ...]   (prints one row per file)"""
import sys
import numpy as np, h5py

GOFF, N, wrap = 25, 50, lambda d: np.abs((d + np.pi) % (2 * np.pi) - np.pi)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    fl, ep, Q = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:]
cs = np.r_[0, np.cumsum(fl)]; t = np.arange(len(fl) - GOFF)
ok = (ep[t] >= 9000) & (ep[t] == ep[t + GOFF]) & (fl[t] == 0) & (fl[t + GOFF] == 0) & (cs[t + GOFF + 1] - cs[t] > 0)
cand = t[ok]; sel = np.sort(cand[np.random.default_rng(42).choice(len(cand) - 1, size=N, replace=False)])
GOALS = Q[sel + GOFF]


def outcomes(path):
    d = np.load(path); q, succ, coll = d["q"], d["success"].astype(bool), d["collision"].astype(bool)
    assert np.allclose(q[:, 0], Q[sel], atol=1e-6), f"{path}: starts do not match the data-mode pairs"
    gd = wrap(q - GOALS[:, None]).max(-1)                                   # (N, T+1) goal distance
    prog = 1 - gd.min(1) / np.maximum(gd[:, 0], 1e-6)
    plen = wrap(np.diff(q, axis=1)).max(-1).sum(1)
    sf = ~succ & ~coll
    att = dict(fooled=float("nan"), forced=float("nan"), noticed=float("nan"))
    if "plan_minh" in d.files and coll.any():
        box, pm, hr = d["in_box"], d["plan_minh"], d["h_real"]
        rows = []
        for j in np.where(coll)[0]:
            e = int(np.argmax(box[j])); r = min((e - 1) // (hr.shape[1] // len(pm)), len(pm) - 1)
            rows.append((pm[r, j] > 0, hr[j, e - 1] < 0))
        rows = np.array(rows); att = dict(fooled=rows[:, 0].mean(), forced=1 - rows[:, 0].mean(), noticed=rows[:, 1].mean())
    return dict(**att, SS=(succ & ~coll).mean(), SF=sf.mean(), US=(succ & coll).mean(), UF=(~succ & coll).mean(),
                progress=np.median(prog), path=np.median(plen), stall=(prog[sf] < 0.25).mean() if sf.any() else float("nan"))


if __name__ == "__main__":
    print("run | SS | SF (cannot plan) | US | UF | progress | path | stall (of SF) | fooled | forced | noticed (of collisions)")
    for p in sys.argv[1:]:
        o = outcomes(p)
        print(f"{p.split('/')[-1][:-4]} | {o['SS']:.3f} | {o['SF']:.3f} | {o['US']:.3f} | {o['UF']:.3f} | {o['progress']:.3f} | {o['path']:.2f} | {o['stall']:.3f} | {o['fooled']:.2f} | {o['forced']:.2f} | {o['noticed']:.2f}")
