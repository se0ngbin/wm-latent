"""Self-contained JEPA static margin-OOD (train + eval in ONE process, one
renderer) — the trustworthy version of the split pipeline (ood_margin.sh +
ood_margin_eval.py), which had a train/eval render seam that made the color
oracle inconsistent. Mirrors ood_margin_dreamer.py exactly, so JEPA and Dreamer
numbers are directly comparable.

WM frozen. rk=0 (static; head trained on real encoded frames only). For each
frozen encoder and each obstacle-appearance axis we train a fresh MarginHead
(192-512-512-1, HJ-regression) on the OG appearance (M_og) and on the NEW
appearance (M_new), then report sign-acc + AUC of:
  M_og  on og  (in-dist),  M_og on new (zero-shot),  M_new on new (oracle).

Usage: ood_margin_jepa_static.py <axis>  [seed]     axis in {color,shape,rotate}
"""
import os, sys, math
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw
from scipy.stats import pearsonr
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv
from tworoom_safe_solver import MarginHead
from utils import get_img_preprocessor

DEV, IMG, N, NTR = "cuda:0", 224, 8000, 6000
HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
ENCODERS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
            "jacobian+pull": "lewm_dubins_jacpull50"}
AXES = {
    "color":  (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="purple", shape="circle", rot=0), "red", "purple"),
    "shape":  (dict(obstacle="red", shape="circle", rot=0),
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
        draw.polygon([(cx,cy-rp),(cx+rp,cy),(cx,cy+rp),(cx-rp,cy)], fill=env.colors["obstacle"])
    gc = w2p(env.goal_position); gr = env.goal_radius/(env.x_max-env.x_min)*h[0]
    draw.ellipse([(gc[0]-gr,gc[1]-gr),(gc[0]+gr,gc[1]+gr)], fill=env.colors["goal"])
    env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def render_cond(env, st, cond, tf):
    env.colors = {"background": "white", "agent": "blue", "goal": "green", "obstacle": cond["obstacle"]}
    sq = cond["shape"] == "square"; rot = cond.get("rot", 0)
    imgs = []
    for s in st:
        env.state = s.copy()
        img = render_square(env) if sq else env._render_image()
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


def train_head(Ztr, ytr, seed=0, steps=4000):
    torch.manual_seed(seed)
    head = MarginHead().to(DEV)  # fresh init
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4)
    for _ in range(steps):
        idx = torch.randint(0, Ztr.shape[0], (256,), device=DEV)
        loss = nn.functional.mse_loss(head(Ztr[idx]), ytr[idx])
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def score(head, Z, y):
    with torch.no_grad():
        p = head(Z.to(DEV)).cpu().numpy()
    acc = float(((p >= 0) == (y >= 0)).mean())
    auc = float(roc_auc_score((y >= 0).astype(int), p)) if 0 < (y>=0).mean() < 1 else float("nan")
    return acc, auc


def main():
    axis = sys.argv[1] if len(sys.argv) > 1 else "color"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    og_cond, new_cond, og_tag, new_tag = AXES[axis]
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(seed)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    y = hj_interp(V, xs, ys, ths, st).astype(np.float32)
    yte = y[NTR:]
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    print(f"axis={axis} seed={seed} OG={og_tag} NEW={new_tag} N={N} frac_safe={(y>=0).mean():.3f}")

    for ename, ck in ENCODERS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        Z = {og_tag: encode(m, render_cond(env, st, og_cond, tf)),
             new_tag: encode(m, render_cond(env, st, new_cond, tf))}
        del m; torch.cuda.empty_cache()
        yt = torch.tensor(y[:NTR], device=DEV)
        heads = {t: train_head(Z[t][:NTR].to(DEV), yt, seed=seed) for t in (og_tag, new_tag)}
        print(f"\n=== jepa:{ename}  axis={axis} seed={seed} ===")
        print(f"{'margin / eval-on':26s} {'sign-acc':>9s} {'AUC':>7s}")
        v = {}
        for ht in (og_tag, new_tag):
            for zt in (og_tag, new_tag):
                acc, auc = score(heads[ht], Z[zt][NTR:], yte); v[(ht, zt)] = (acc, auc)
                tagm = {"og_og":"(in-dist)","og_new":"(zero-shot)","new_new":"(oracle)"}.get(
                    f"{'og' if ht==og_tag else 'new'}_{'og' if zt==og_tag else 'new'}", "")
                print(f"M_{ht:<8s} on {zt:<10s}    {acc:9.3f} {auc:7.3f}  {tagm}")
        (oa_o, ou_o) = v[(og_tag, og_tag)]; (za, zu) = v[(og_tag, new_tag)]; (na, nu) = v[(new_tag, new_tag)]
        print(f"KEY jepa {ename} axis={axis} rk=0 seed={seed} "
              f"ogog_acc={oa_o:.3f} ogog_auc={ou_o:.3f} "
              f"zs_acc={za:.3f} zs_auc={zu:.3f} or_acc={na:.3f} or_auc={nu:.3f}")
    print("\ndone")


if __name__ == "__main__":
    main()
