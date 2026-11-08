"""Save a label-free test-time BatchNorm-adapted copy of a Dubins LeWM for one condition (the procedure screened by
diag_imag_fs_dubins.py BNADAPT=1): projector BN statistics recomputed from NADAPT unlabeled frames of the condition
(random uniform states), then the predictor output projection's BN statistics from NB-step random-action imagined rollouts
from those latents. Nothing else changes. Writes checkpoints/<wm>_bn<cond>/weights_epoch_50.pt (+ config links), so CEM
loads it like any WM; the margin function h stays the red-trained one of <wm>.
Env: NADAPT (2000), SAFEONLY (0/1), BNTAG (suffix of the output dir).
Usage: bn_adapt_save.py <wm_dir> <cond: purple|rot90|red> [fs=1] [nb=25]"""
import sys, os, copy
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..")); sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from safe_solver import _load_margin_head   # noqa: F401  (import order, see diag_imag_color_dubins.py)
import ood_margin_gp_jepa as E
import stable_worldmodel as swm

ck, cn = sys.argv[1], sys.argv[2]
FS = int(sys.argv[3]) if len(sys.argv) > 3 else 1
NB = int(sys.argv[4]) if len(sys.argv) > 4 else 25
NADAPT, DEV = int(os.environ.get("NADAPT", "2000")), E.DEV
COND = {"red": dict(obstacle="red", shape="circle", rot=0), "purple": dict(obstacle="purple", shape="circle", rot=0),
        "rot90": dict(obstacle="red", shape="circle", rot=90)}[cn]
tf = E.get_img_preprocessor(source="pixels", target="pixels", img_size=E.IMG)
tf_fixed = lambda d: tf({"pixels": torch.from_numpy(np.ascontiguousarray(d["pixels"])).permute(0, 3, 1, 2)})
ra = np.random.default_rng(11)                                      # same adaptation set as the screen
x0, x1, y0, y1 = E.WORLD_BOUNDS
SAFEONLY = bool(int(os.environ.get("SAFEONLY", "0")))   # adaptation frames only of SAFE states (a robot cannot collect collisions)
sa = np.zeros((0, 3), np.float32)
while len(sa) < NADAPT:
    c = np.stack([ra.uniform(x0, x1, 2 * NADAPT), ra.uniform(y0, y1, 2 * NADAPT), ra.uniform(-np.pi, np.pi, 2 * NADAPT)], 1).astype(np.float32)
    if SAFEONLY:
        c = c[~E.inside_obstacle(c[:, :2], "circle").astype(bool)]
    sa = np.concatenate([sa, c])
sa = sa[:NADAPT]
Aa = torch.from_numpy((ra.standard_normal((NADAPT, 1, 1)) + 0.7 * ra.standard_normal((NADAPT, NB * FS, 1)))
                      .astype(np.float32).reshape(NADAPT, NB, FS)).to(DEV)
m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
frames = E.render_cond(E.DubinsSwmEnv(), sa, COND, tf_fixed)


def stage(mod):
    bns = [b for b in mod.modules() if isinstance(b, torch.nn.modules.batchnorm._BatchNorm)]
    for b in bns:
        b.reset_running_stats(); b.momentum = None; b.train()
    return bns


with torch.no_grad():
    bns = stage(m.projector)
    E.encode(m, frames, bs=256)
    for b in bns: b.eval()
    z = E.encode(m, frames).to(DEV).unsqueeze(1)
    bns = stage(m.pred_proj)
    for i in range(0, NADAPT, 256):
        e = z[i:i + 256]
        for k in range(NB):
            e = torch.cat([e, m.predict(e[:, -3:], m.action_encoder(Aa[i:i + 256, :k + 1])[:, -3:])[:, -1:]], 1)
    for b in bns: b.eval()
root = os.environ["STABLEWM_HOME"] + "/checkpoints"
src = os.path.realpath(f"{root}/{ck}")
out = f"{root}/{ck}_bn{cn}" + os.environ.get("BNTAG", "")
os.makedirs(out, exist_ok=True)
torch.save(m.state_dict(), f"{out}/weights_epoch_50.pt")
for c in ("config.json", "config.yaml"):
    if os.path.exists(f"{src}/{c}") and not os.path.exists(f"{out}/{c}"):
        os.symlink(f"{src}/{c}", f"{out}/{c}")
print(f"saved {out}/weights_epoch_50.pt", flush=True)
