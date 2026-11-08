"""Imagined-state safety screen for any frameskip (diag_imag_color_dubins.py generalized), plus how often block-end checks
miss a collision. 600 held-out expert start states (last 10% of episodes), rendered red / purple / rot 90 (training
orientation otherwise); random CEM-style plans of NB blocks x FS env steps (per-step actions z-scored like training,
stacked FS per block as the action encoder takes them); simulator labels at EVERY env step. Prints per condition and h:
  AUC of h on the imagined latent at each block end vs the true block-end label (env depth = block * FS)
  missed: share of plans with a collision strictly inside a block while every block end is safe (invisible to CCEM's
          block-end checks), and the share of all colliding plans that are such misses.
BNADAPT=1: label-free test-time BatchNorm adaptation per condition, on a fresh copy of the WM: (1) the projector's BN
statistics are recomputed from NADAPT unlabeled frames of the condition (random uniform states, disjoint from the test
starts); (2) the predictor output projection's BN statistics from NB-block imagined rollouts (random actions) started at
those adapted latents. h stays the red-trained one; red gets the same treatment as a control. Also prints real-frame AUC.
Usage: diag_imag_fs_dubins.py <wm_dir> <fs> <n_blocks> <head.pt> [...]"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch, h5py
from sklearn.metrics import roc_auc_score
from safe_solver import _load_margin_head   # before E (import order, see diag_imag_color_dubins.py)
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, FS, NB, heads = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4:]
N, DEV = 600, E.DEV
rng = np.random.default_rng(2)
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
with h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r") as f:
    off, ln = f["ep_offset"][:], f["ep_len"][:]
    act_all = f["action"][:].astype(np.float32).reshape(len(f["action"]), -1)
    a_mu, a_sd = np.nanmean(act_all, 0), np.nanstd(act_all, 0)
    e = rng.choice(np.arange(int(0.9 * len(ln)), len(ln)), N)
    st0 = f["state"][:][off[e] + rng.integers(0, ln[e])]
T = NB * FS
A = (rng.standard_normal((N, 1, 1)) + 0.7 * rng.standard_normal((N, T, 1))).astype(np.float32)   # per env step
env, y = E.DubinsSwmEnv(), np.zeros((N, T), bool)
for i in range(N):
    env.state = st0[i].astype(np.float32).copy()
    for k in range(T):
        env._integrate(float(np.clip(A[i, k, 0] * a_sd[0] + a_mu[0], -env.max_angular_velocity, env.max_angular_velocity)))
        y[i, k] = bool(E.inside_obstacle(env.state[None, :2], "circle")[0])
yb = y[:, FS - 1::FS]                                            # block-end labels (N, NB)
mid = y.reshape(N, NB, FS)[:, :, :-1].any(-1)                    # collision strictly inside a block
miss = mid.any(1) & ~yb.any(1)
print(f"MISS {ck} fs{FS}: plans colliding mid-block with all block ends safe {miss.mean():.3f}; "
      f"of colliding plans {miss.sum() / max(1, y.any(1).sum()):.3f}", flush=True)
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
Hs = [_load_margin_head(p, DEV).eval() for p in heads]
Ab = torch.from_numpy(A.reshape(N, NB, FS)).to(DEV)              # FS actions stacked per block
CONDS = {"red": dict(obstacle="red", shape="circle", rot=0), "purple": dict(obstacle="purple", shape="circle", rot=0),
         "rot90": dict(obstacle="red", shape="circle", rot=90)}
BNADAPT, NADAPT = bool(int(os.environ.get("BNADAPT", "0"))), int(os.environ.get("NADAPT", "2000"))
ra = np.random.default_rng(11)
x0, x1, y0, y1 = E.WORLD_BOUNDS
sa = np.stack([ra.uniform(x0, x1, NADAPT), ra.uniform(y0, y1, NADAPT), ra.uniform(-np.pi, np.pi, NADAPT)], 1).astype(np.float32)
Aa = torch.from_numpy((ra.standard_normal((NADAPT, 1, 1)) + 0.7 * ra.standard_normal((NADAPT, T, 1))).astype(np.float32).reshape(NADAPT, NB, FS)).to(DEV)
m0 = m


def bn_adapt(frames):
    """Recompute BatchNorm statistics (cumulative average, no labels) on the condition's frames, then on rollouts."""
    m = copy.deepcopy(m0).eval()
    def stage(mod):
        bns = [b for b in mod.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
        for b in bns:
            b.reset_running_stats(); b.momentum = None; b.train()
        return bns
    with torch.no_grad():
        bns = stage(m.projector)
        E.encode(m, frames, bs=256)                                      # projector BN <- condition frames
        for b in bns: b.eval()
        z = E.encode(m, frames).to(DEV).unsqueeze(1)
        bns = stage(m.pred_proj)
        for i in range(0, NADAPT, 256):                                  # pred_proj BN <- imagined rollouts
            e = z[i:i + 256]
            for b in range(NB):
                e = torch.cat([e, m.predict(e[:, -3:], m.action_encoder(Aa[i:i + 256, :b + 1])[:, -3:])[:, -1:]], 1)
        for b in bns: b.eval()
    return m


for cn, cond in CONDS.items():
    if BNADAPT:
        m = bn_adapt(E.render_cond(E.DubinsSwmEnv(), sa, cond, tf_fixed))
    with torch.no_grad():
        emb = E.encode(m, E.render_cond(E.DubinsSwmEnv(), st0, cond, tf_fixed)).to(DEV).unsqueeze(1)
        for b in range(NB):
            emb = torch.cat([emb, m.predict(emb[:, -3:], m.action_encoder(Ab[:, :b + 1])[:, -3:])[:, -1:]], 1)
        Zh = emb[:, 1:]
    for p, hd in zip(heads, Hs):
        with torch.no_grad():
            h = hd(Zh.flatten(0, 1).float()).view(N, NB).cpu().numpy()
            hr = hd(emb[:, 0].float()).cpu().numpy()
        safe = ~yb
        y0l = np.array([bool(E.inside_obstacle(st0[i:i + 1, :2], "circle")[0]) for i in range(N)])
        real = f"real {roc_auc_score(~y0l, hr):.2f} " if 0 < y0l.mean() < 1 else ""
        aucs = " ".join(f"{(b + 1) * FS}:{roc_auc_score(safe[:, b], h[:, b]):.2f}" for b in range(NB) if 0 < safe[:, b].mean() < 1)
        print(f"IMAGFS{'-BN' if BNADAPT else ''} {ck} fs{FS} {os.path.basename(p)} {cn}: {real}AUC@envstep {aucs}", flush=True)
