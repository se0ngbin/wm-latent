"""Evaluate margin heads (safe/unsafe CBF classifiers) under obstacle-appearance
shift. For each encoder we have two margin functions: one trained on the ORIGINAL
obstacle appearance (M_og) and one retrained on the NEW appearance (M_new). We
score BOTH on the NEW-appearance test states (and on OG, for reference) vs the HJ
ground truth V*.

  M_og  on new  = does a margin trained on the original look survive the shift?
  M_new on new  = retrain-for-new ceiling (info recoverable?).
Compared across baseline / jacobian / jacobian+pull encoders (WM frozen).

Geometry is identical across appearances, so HJ V* labels stay exact (circle
collision even when the obstacle is drawn as a diamond).

Usage: ood_margin_eval.py <axis>   axis in {color, shape}
"""
import os, sys, math
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch
from PIL import Image, ImageDraw
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv
from tworoom_safe_solver import MarginHead
from utils import get_img_preprocessor

DEV, IMG, N = "cuda:0", 224, 8000
HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
MDIR = "/data/seongbin/dreamer/lewm"
ENCODERS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
            "jacobian+pull": "lewm_dubins_jacpull50"}

AXES = {
    # axis -> (og_cond, new_cond, og_tag, new_tag); tag names the trained head file
    "color": (dict(obstacle="red",    shape="circle", rot=0),
              dict(obstacle="purple", shape="circle", rot=0), "red", "purple"),
    "shape": (dict(obstacle="red", shape="circle", rot=0),
              dict(obstacle="red", shape="square", rot=0), "circle", "square"),
    "rotate": (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="circle", rot=90), "rot0", "rot90"),
}


def hj_interp(V, xs, ys, ths, pts):
    def idxw(g, v, periodic=False):
        if periodic: v = (v - g[0]) % (2*np.pi) + g[0]
        i = np.clip(np.searchsorted(g, v) - 1, 0, len(g)-2)
        return i, np.clip((v - g[i]) / (g[i+1]-g[i]+1e-9), 0, 1)
    ix, wx = idxw(xs, pts[:,0]); iy, wy = idxw(ys, pts[:,1]); it, wt = idxw(ths, pts[:,2], True)
    out = np.zeros(len(pts))
    for dx in (0,1):
        for dy in (0,1):
            for dt in (0,1):
                w = (wx if dx else 1-wx)*(wy if dy else 1-wy)*(wt if dt else 1-wt)
                out += w * V[np.clip(ix+dx,0,len(xs)-1), np.clip(iy+dy,0,len(ys)-1), (it+dt)%len(ths)]
    return out


def render_square(env):
    scale = 4; h = (env.image_size[0]*scale, env.image_size[1]*scale)
    img = Image.new("RGB", h, env.colors["background"]); draw = ImageDraw.Draw(img)
    def w2p(c):
        x, y = c
        return (int((x-env.x_min)/(env.x_max-env.x_min)*h[0]),
                int((env.y_max-y)/(env.y_max-env.y_min)*h[1]))
    for ox, oy, r in env.obstacles:
        cx, cy = w2p((ox, oy)); rp = r/(env.x_max-env.x_min)*h[0]
        draw.polygon([(cx, cy-rp), (cx+rp, cy), (cx, cy+rp), (cx-rp, cy)], fill=env.colors["obstacle"])
    gc = w2p(env.goal_position); gr = env.goal_radius/(env.x_max-env.x_min)*h[0]
    draw.ellipse([(gc[0]-gr, gc[1]-gr), (gc[0]+gr, gc[1]+gr)], fill=env.colors["goal"])
    env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def render_cond(env, st, cond, tf):
    env.colors = {"background": "white", "agent": "blue", "goal": "green",
                  "obstacle": cond["obstacle"]}
    rot = cond.get("rot", 0)
    imgs = []
    for s in st:
        env.state = s.copy()
        img = render_square(env) if cond["shape"] == "square" else env._render_image()
        if rot:
            img = np.ascontiguousarray(np.rot90(img, k=rot // 90))
        imgs.append(tf({"pixels": img[None]})["pixels"][0])
    return torch.stack(imgs).float()


def encode(m, frames, bs=64):
    Z = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], bs):
            out = m.encoder(frames[i:i+bs].to(DEV), interpolate_pos_encoding=True)
            Z.append(m.projector(out.last_hidden_state[:, 0]).cpu())
    return torch.cat(Z)


def score(head, Z, y):
    with torch.no_grad():
        p = head(Z.to(DEV)).cpu().numpy()
    acc = float(((p >= 0) == (y >= 0)).mean())
    auc = float(roc_auc_score((y >= 0).astype(int), p)) if 0 < (y>=0).mean() < 1 else float("nan")
    r = float(pearsonr(p, y)[0])
    return acc, auc, r


def main():
    # argv: axis [rollout_k] [seed]
    axis = sys.argv[1] if len(sys.argv) > 1 else "color"
    rk = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    seed = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    suffix = f"_rk{rk}_s{seed}"
    og_cond, new_cond, og_tag, new_tag = AXES[axis]
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(1000 + seed)  # eval states independent per seed
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = hj_interp(V, xs, ys, ths, st).astype(np.float32)
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    print(f"axis={axis} rk={rk} seed={seed}  OG={og_tag} NEW={new_tag}  N={N} frac_safe={(y>=0).mean():.3f}")

    for ename, ck in ENCODERS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        Z_og = encode(m, render_cond(env, st, og_cond, tf))
        Z_new = encode(m, render_cond(env, st, new_cond, tf))
        del m; torch.cuda.empty_cache()
        heads = {}
        for tag in (og_tag, new_tag):
            hp = os.path.join(MDIR, ck, f"margin_hj_{axis}_{tag}{suffix}.pt")
            hd = MarginHead().to(DEV); hd.load_state_dict(torch.load(hp, map_location=DEV)["margin_gp"]); hd.eval()
            heads[tag] = hd
        print(f"\n=== {ename} ===")
        print(f"{'margin / eval-on':28s} {'sign-acc':>9s} {'AUC':>7s} {'pearson':>8s}")
        vals = {}
        for htag in (og_tag, new_tag):
            for ztag, Z in ((og_tag, Z_og), (new_tag, Z_new)):
                acc, auc, r = score(heads[htag], Z, y)
                vals[(htag, ztag)] = (acc, auc)
                mark = "  <- M_%s on NEW" % htag if ztag == new_tag else ""
                print(f"M_{htag:<8s} on {ztag:<12s}      {acc:9.3f} {auc:7.3f} {r:8.3f}{mark}")
        # machine-readable summary: zero-shot (M_og on NEW) + oracle (M_new on NEW)
        zs_acc, zs_auc = vals[(og_tag, new_tag)]
        or_acc, or_auc = vals[(new_tag, new_tag)]
        print(f"KEY jepa {ename} axis={axis} rk={rk} seed={seed} "
              f"zs_acc={zs_acc:.3f} zs_auc={zs_auc:.3f} or_acc={or_acc:.3f} or_auc={or_auc:.3f}")
    print("\ndone")


if __name__ == "__main__":
    main()
