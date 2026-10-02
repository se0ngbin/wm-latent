"""Wall-clearance analysis for TwoRoom (analog of dubins obstacle_dist).

Computes, per planned trajectory, the agent's distance (px) to the dividing wall
(a hard obstacle with one door gap). Reports mean/median closest approach and mean
clearance, baseline vs jacobian. Reads npz trajectory dumps produced by running
eval.py with traj_patch_tworoom imported.

Usage: python tworoom_wall_dist.py  (reads outputs/wallobsdist_{base,jac}.npz)
"""
import numpy as np
from scipy.ndimage import distance_transform_edt

IMG, WALL_X, HALF, DOOR_Y0, DOOR_Y1, BORDER = 224, 112, 5, 35, 63, 14
solid = np.zeros((IMG, IMG), bool)                       # [y,x]
solid[BORDER:IMG-BORDER, WALL_X-HALF:WALL_X+HALF+1] = True
solid[DOOR_Y0:DOOR_Y1, WALL_X-HALF:WALL_X+HALF+1] = False
FREE_DIST = distance_transform_edt(~solid)

def wall_dist(pts):
    xi = np.clip(pts[:, 0].astype(int), 0, IMG-1); yi = np.clip(pts[:, 1].astype(int), 0, IMG-1)
    d = FREE_DIST[yi, xi]
    return np.where(solid[yi, xi], -1.0, d)

if __name__ == "__main__":
    import sys
    rows = [("baseline", "outputs/wallobsdist_base.npz"),
            ("jacobian", "outputs/wallobsdist_jac.npz")]
    print(f"{'model':12s} {'n':>4s} {'mean_min':>9s} {'median_min':>11s} {'mean_clear':>11s} {'%within3px':>10s}")
    for lab, f in rows:
        try:
            z = np.load(f)
        except FileNotFoundError:
            print(f"{lab:12s}  MISSING {f}"); continue
        trajs = [z[k] for k in z.files]
        mins = np.array([wall_dist(t).min() for t in trajs])
        means = np.array([wall_dist(t).mean() for t in trajs])
        print(f"{lab:12s} {len(trajs):4d} {mins.mean():+9.2f} {np.median(mins):+11.2f} "
              f"{means.mean():+11.2f} {100*np.mean(mins<3):9.1f}%")
