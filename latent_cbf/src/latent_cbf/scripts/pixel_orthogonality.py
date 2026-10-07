"""Premise test for the conditioning->orthogonality argument: is the appearance shift
ORTHOGONAL to the safety axis in PIXEL space (before any encoder)? If yes, a well-
conditioned (angle-preserving) encoder would keep them orthogonal in latent; an ill-
conditioned one entangles them. Pixel-space:
  mu_pix   = mean_states( img(shift) - img(red) )       (the recolor / rotate direction)
  la_pix   = mean(img[unsafe]) - mean(img[safe])        (failure-label safety direction)
  cos_pix  = |cos(mu_pix, la_pix)|
Compare cos_pix (input) to cos_la (latent, per encoder): entanglement = cos_la >> cos_pix."""
import sys
from pathlib import Path
import numpy as np
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
from configs import Config
from critic_ood_eval import render_cond, CONDS
from latent_cbf.dubins.dubins_env import DubinsEnv

N = 2000

def cos(a, b):
    return float(abs((a@b)/(np.linalg.norm(a)*np.linalg.norm(b)+1e-12)))

def flat(imgs):  # imgs: (N,H,W,3) uint8 (or list) -> (N, H*W*3) float
    a = np.asarray(imgs, dtype=np.float32)
    return a.reshape(a.shape[0], -1)

def main():
    ec = Config().environment
    obst = ec.get_obstacles_list()
    env = DubinsEnv(image_size=tuple(ec.image_size), world_bounds=tuple(ec.world_bounds),
                    max_angular_velocity=ec.max_angular_velocity, speed=ec.speed, dt=ec.dt,
                    obstacles=obst, goal_radius=ec.goal_radius,
                    collision_radius=ec.collision_radius, render_mode="rgb_array")
    rng = np.random.default_rng(0); xs = ec.world_bounds
    st = np.stack([rng.uniform(xs[0], xs[1], N), rng.uniform(xs[2], xs[3], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    unsafe = np.zeros(N, int)
    for cx, cy, r in obst:
        unsafe |= (np.hypot(st[:,0]-cx, st[:,1]-cy) <= r).astype(int)
    Xr = flat(render_cond(env, st, CONDS["red"]))
    la_pix = Xr[unsafe==1].mean(0) - Xr[unsafe==0].mean(0)     # safety direction in pixels
    for axis, cname in (("color","purple"), ("rotate","rot90")):
        Xs = flat(render_cond(env, st, CONDS[cname]))
        mu_pix = (Xs - Xr).mean(0)
        # how many pixels each direction touches (support), for intuition
        supp_mu = int((np.abs(mu_pix) > 1.0).sum()); supp_la = int((np.abs(la_pix) > 1.0).sum())
        overlap = int(((np.abs(mu_pix) > 1.0) & (np.abs(la_pix) > 1.0)).sum())
        print(f"PIX {axis:6s} cos_pix={cos(mu_pix, la_pix):.3f}  "
              f"|mu supp|={supp_mu} |la supp|={supp_la} overlap={overlap} "
              f"(frac_unsafe={unsafe.mean():.3f})")
    print("PIX_DONE")

if __name__ == "__main__":
    main()
