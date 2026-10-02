"""OOD classifier-transfer on frozen LE-WM encoders (Thread C1+C2, static).

Question: does an invariance-regularized (jacobian) encoder keep its safe/unsafe
structure under APPEARANCE shift better than the baseline (sigreg) encoder?

Setup — no world-model retraining, everything we already have:
  * Dubins geometry (obstacles, world bounds) is UNCHANGED across conditions, so
    the HJ avoid value V*(x,y,theta) labels stay EXACT for every appearance.
  * We only perturb rendering: obstacle color, background color, agent color,
    and (flagged) obstacle shape. These are pure nuisance for the safe set.
  * For each frozen encoder we train ONE head on the in-distribution frames, then
    score that same head on each shifted condition. Degradation = in-dist minus
    OOD sign-acc/AUC. Small degradation for jacobian vs large for baseline = the
    invariance payoff, measured.

Heads: linear + a small MLP (does capacity change the transfer story?).
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import math
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv, OBSTACLES, WORLD_BOUNDS, IMAGE_SIZE
from utils import get_img_preprocessor

DEV, IMG, N, NTR = "cuda:0", 224, 8000, 6000
HJ = "/home/seongbin/latent/latent_cbf/results/hj_truth.npz"
MODELS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
          "jacobian+pull": "lewm_dubins_jacpull50"}
HEADS = {"linear": [], "mlp-512x2": [512, 512]}

# Appearance conditions. geometry (obstacles/bounds) identical -> V* labels exact.
# 'shape' redraws the obstacle as a 45-deg square of half-width r (bounding the
# circle) -> mild geometry change at the corners; flagged, not label-exact.
CONDITIONS = {
    "in-dist":     dict(background="white", agent="blue",  goal="green", obstacle="red",    shape="circle"),
    "obst-purple": dict(background="white", agent="blue",  goal="green", obstacle="purple", shape="circle"),
    "obst-orange": dict(background="white", agent="blue",  goal="green", obstacle="orange", shape="circle"),
    "bg-gray":     dict(background=(200,200,200), agent="blue", goal="green", obstacle="red", shape="circle"),
    "bg-tan":      dict(background=(210,190,150), agent="blue", goal="green", obstacle="red", shape="circle"),
    "agent-black": dict(background="white", agent="black", goal="green", obstacle="red",     shape="circle"),
    "all-shift":   dict(background=(210,190,150), agent="black", goal="green", obstacle="purple", shape="circle"),
    "shape-square":dict(background="white", agent="blue",  goal="green", obstacle="red",     shape="square"),
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


def render_state(env, state, cond, tf):
    """Render one state under an appearance condition (patches env.colors + shape)."""
    env.state = state.copy()
    env.colors = {"background": cond["background"], "agent": cond["agent"],
                  "goal": cond["goal"], "obstacle": cond["obstacle"]}
    if cond["shape"] == "circle":
        img = env._render_image()
    else:
        img = _render_square_obstacle(env)
    return tf({"pixels": img[None]})["pixels"][0]


def _render_square_obstacle(env):
    """Same renderer as DubinsSwmEnv._render_image but obstacles are 45-deg squares
    (half-width r) instead of circles. Appearance/shape nuisance; mild label change."""
    scale = 4
    h = (env.image_size[0]*scale, env.image_size[1]*scale)
    img = Image.new("RGB", h, env.colors["background"]); draw = ImageDraw.Draw(img)
    def w2p(c):
        x, y = c
        return (int((x-env.x_min)/(env.x_max-env.x_min)*h[0]),
                int((env.y_max-y)/(env.y_max-env.y_min)*h[1]))
    for ox, oy, r in env.obstacles:
        cx, cy = w2p((ox, oy)); rp = r/(env.x_max-env.x_min)*h[0]
        pts = [(cx, cy-rp), (cx+rp, cy), (cx, cy+rp), (cx-rp, cy)]  # diamond
        draw.polygon(pts, fill=env.colors["obstacle"])
    gc = w2p(env.goal_position); gr = env.goal_radius/(env.x_max-env.x_min)*h[0]
    draw.ellipse([(gc[0]-gr, gc[1]-gr), (gc[0]+gr, gc[1]+gr)], fill=env.colors["goal"])
    env._draw_agent(draw, w2p(env.state[:2]), float(env.state[2]), scale)
    return np.array(img.resize(env.image_size, Image.Resampling.LANCZOS))


def encode(m, frames, bs=64):
    Z = []
    with torch.no_grad():
        for i in range(0, frames.shape[0], bs):
            out = m.encoder(frames[i:i+bs].to(DEV), interpolate_pos_encoding=True)
            Z.append(m.projector(out.last_hidden_state[:, 0]).cpu())
    return torch.cat(Z)


def main():
    d = np.load(HJ); V, xs, ys, ths = d["V"], d["grid_xs"], d["grid_ys"], d["grid_thetas"]
    rng = np.random.default_rng(0)
    st = np.stack([rng.uniform(xs[0], xs[-1], N), rng.uniform(ys[0], ys[-1], N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    vstar = hj_interp(V, xs, ys, ths, st).astype(np.float32)
    y = torch.tensor(vstar)
    print(f"N={N} train={NTR}  frac_safe(V*>=0)={(vstar>=0).mean():.3f}")

    env = DubinsSwmEnv()
    tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)

    # Pre-render every condition's frames once (shared across models).
    print("rendering conditions...")
    frames_by_cond = {}
    for cname, cond in CONDITIONS.items():
        imgs = [render_state(env, st[i], cond, tf) for i in range(N)]
        frames_by_cond[cname] = torch.stack(imgs).float()
        print(f"  {cname:13s} rendered")

    yte = y[NTR:].numpy()
    for mname, ck in MODELS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        Z = {c: encode(m, f) for c, f in frames_by_cond.items()}
        del m; torch.cuda.empty_cache()
        # standardize with in-dist train stats (applied to every condition)
        mu, sd = Z["in-dist"][:NTR].mean(0), Z["in-dist"][:NTR].std(0) + 1e-6
        Zn = {c: (z - mu) / sd for c, z in Z.items()}
        for hname, hid in HEADS.items():
            torch.manual_seed(0)
            layers, dp = [], Z["in-dist"].shape[1]
            for hh in hid: layers += [nn.Linear(dp, hh), nn.SiLU()]; dp = hh
            layers += [nn.Linear(dp, 1)]
            head = nn.Sequential(*layers).to(DEV)
            opt = torch.optim.Adam(head.parameters(), lr=1e-3, weight_decay=1e-4)
            Ztr = Zn["in-dist"][:NTR].to(DEV); ytr = y[:NTR].to(DEV)
            for _ in range(3000):
                idx = torch.randint(0, NTR, (512,), device=DEV)
                loss = nn.functional.mse_loss(head(Ztr[idx]).squeeze(-1), ytr[idx])
                opt.zero_grad(); loss.backward(); opt.step()
            head.eval()

            def train_head(Ztr_c, ytr_c):
                torch.manual_seed(0)
                ll, dpp = [], Z["in-dist"].shape[1]
                for hh in hid: ll += [nn.Linear(dpp, hh), nn.SiLU()]; dpp = hh
                ll += [nn.Linear(dpp, 1)]
                hd = nn.Sequential(*ll).to(DEV)
                op = torch.optim.Adam(hd.parameters(), lr=1e-3, weight_decay=1e-4)
                for _ in range(3000):
                    ix = torch.randint(0, Ztr_c.shape[0], (512,), device=DEV)
                    lo = nn.functional.mse_loss(hd(Ztr_c[ix]).squeeze(-1), ytr_c[ix])
                    op.zero_grad(); lo.backward(); op.step()
                return hd.eval()

            base_acc = None
            # transfer: head trained on in-dist, scored on each condition.
            # oracle:  a fresh head trained ON that condition (retrain-on-OOD, C2b) ->
            #          separates recoverable calibration shift from info collapse.
            print(f"\n[{mname} | {hname}]  {'condition':13s} {'xfer-acc':>8s} {'xfer-AUC':>8s} "
                  f"{'d-acc':>6s} {'oracle-acc':>10s}")
            for cname in CONDITIONS:
                with torch.no_grad():
                    p = head(Zn[cname][NTR:].to(DEV)).squeeze(-1).cpu().numpy()
                acc = ((p >= 0) == (yte >= 0)).mean()
                auc = roc_auc_score((yte >= 0).astype(int), p)
                if base_acc is None: base_acc = acc
                dd = acc - base_acc
                if cname == "in-dist":
                    orc = acc
                else:
                    oh = train_head(Zn[cname][:NTR].to(DEV), y[:NTR].to(DEV))
                    with torch.no_grad():
                        po = oh(Zn[cname][NTR:].to(DEV)).squeeze(-1).cpu().numpy()
                    orc = ((po >= 0) == (yte >= 0)).mean()
                flag = " (mild-geom)" if cname == "shape-square" else ""
                print(f"{'':22s} {cname:13s} {acc:8.3f} {auc:8.3f} {dd:+6.3f} {orc:10.3f}{flag}")
    print("\ndone")


if __name__ == "__main__":
    main()
