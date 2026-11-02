"""Why does soft-cost CEM still collide? Splits every collided episode by what the executed plan predicted at the
moment of the collision (trace written by soft_cem.sh with TRACE=<dir>):
  predicted unsafe: h(z_hat) < 0 at or before the collision step in the plan being executed -> the planner chose a plan
                    it knew violates (no predicted-safe plan found);
  false-safe:       h(z_hat) >= 0 up to the collision step -> the world model / margin function missed it.
Replan k of an episode starts after k*REC env steps; imagined step j of that plan is env step k*REC + j + 1.
Usage: diag_collision_cause.py <trace_prefix> [...]   (no args: self-check)"""
import sys
import numpy as np
import torch


def split(trace, outcome, rec):
    per_ep = {}                                           # episode -> list of plan h arrays, in replan order
    for idx, h in trace:
        for i, row in zip(idx, h):
            per_ep.setdefault(int(i), []).append(row)
    n_unsafe = n_false = 0
    for e in np.flatnonzero(outcome["collided"]):
        t = int(outcome["t_collide"][e]); k = (t - 1) // rec; j = t - k * rec - 1
        plans = per_ep.get(e, [])
        if k >= len(plans):                               # no replan covers the step (should not happen)
            continue
        if (plans[k][:j + 1] < 0).any():
            n_unsafe += 1
        else:
            n_false += 1
    return n_unsafe, n_false


def report(prefix):
    o = np.load(prefix + "_outcome.npz"); tr = torch.load(prefix + "_plan_h.pt", weights_only=False)
    rec = 5 if "_rec5" in prefix else 25
    u, f = split(tr, o, rec); n = len(o["collided"])
    print(f"{prefix.rsplit('/', 1)[-1]}: collision {o['collided'].mean() * 100:.1f}% = predicted unsafe "
          f"{u / n * 100:.1f}% + false-safe {f / n * 100:.1f}%  (of collisions: {f / max(u + f, 1) * 100:.0f}% false-safe)")


if __name__ == "__main__":
    if len(sys.argv) == 1:
        # 3 episodes, REC 5: ep0 collides at step 7 (replan 1, imagined 1) with h<0 there -> predicted unsafe;
        # ep1 collides at step 3 with h>=0 up to it -> false-safe; ep2 never collides.
        h0 = np.ones((3, 25)); h1 = np.ones((3, 25)); h1[0, 1] = -0.1; h1[1, 5] = -0.1   # ep1's dip is after its step
        o = {"collided": np.array([1, 1, 0], bool), "t_collide": np.array([7, 3, -1])}
        assert split([(np.arange(3), h0), (np.arange(3), h1)], o, 5) == (1, 1)
        print("ok")
    for p in sys.argv[1:]:
        report(p)
