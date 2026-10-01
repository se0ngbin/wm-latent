"""OOD study on the REAL instantaneous margin function (JEPA), the original
latent_cbf `margin_gp` object — NOT the V*-regression shortcut.

margin_gp(z): a WGAN-GP signed-distance separator of safe vs unsafe features.
  * Label = binary instantaneous `failure`: 1 = colliding (agent INSIDE an
    obstacle), 0 = outside. Computed by geometry per rendered shape, so it is
    LABEL-EXACT for circle / diamond / rotated (no HJ PDE).
  * Loss = repo's _margin_gp_loss: zs_weight*(mean(unsafe)-mean(safe))
           + relu_weight*(relu(unsafe)+relu(-safe)) + gp_weight*(||grad||-thr)^2
    (weights from configs/dreamer_conf.py). Sign>=0 = safe.

WM frozen. Per encoder, train margin_gp on the OG appearance and on the NEW
appearance; report sign-acc + AUC of:
  M_og on og (in-dist), M_og on new (zero-shot), M_new on new (retrained).
Usage: ood_margin_gp_jepa.py <axis> [seed]     axis in {color,shape,rotate}
"""
import os, sys
sys.path.insert(0, "/home/seongbin/latent/le-wm")
os.environ.setdefault("STABLEWM_HOME", "/data/seongbin/lewm")
os.environ.setdefault("LOCAL_DATASET_DIR", "/data/seongbin/lewm")
import numpy as np, torch, torch.nn as nn
from PIL import Image, ImageDraw
from sklearn.metrics import roc_auc_score
import stable_worldmodel as swm
from dubins_swm_env import DubinsSwmEnv, OBSTACLES, WORLD_BOUNDS
from tworoom_safe_solver import MarginHead
from utils import get_img_preprocessor

DEV, IMG, N, NTR = "cuda:0", 224, 8000, 6000
ENCODERS = {"baseline": "sigreg_only_dubins", "jacobian": "jacobian_w1_dubins",
            "jacobian+pull": "lewm_dubins_jacpull50"}
if os.environ.get("ENCODERS"):   # "name:ckpt,name:ckpt" to override (e.g. AC-MTM encoders)
    ENCODERS = dict(kv.split(":") for kv in os.environ["ENCODERS"].split(","))
# margin loss config — configs/dreamer_conf.py (exact)
GRAD_THR, ZS_W, RELU_W, GP_W, GAMMA_LX, LR, STEPS = 0.1, 0.1, 1.0, 10.0, 0.75, 3e-4, 5000
# axis -> (og_cond, new_cond, og_tag, new_tag); cond carries render + label shape
AXES = {
    "color":  (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="purple", shape="circle", rot=0), "red", "purple"),
    "shape":  (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="diamond", rot=0), "circle", "diamond"),
    "rotate": (dict(obstacle="red", shape="circle", rot=0),
               dict(obstacle="red", shape="circle", rot=90), "rot0", "rot90"),
}


def inside_obstacle(xy, shape):
    """Binary failure label: 1 if the agent center is inside an obstacle.
    circle: ||p-c||<=r ; diamond: L1 ball |dx|+|dy|<=r (matches the render)."""
    hit = np.zeros(len(xy), bool)
    for cx, cy, r in OBSTACLES:
        dx, dy = np.abs(xy[:, 0] - cx), np.abs(xy[:, 1] - cy)
        d = (dx + dy) if shape == "diamond" else np.hypot(dx, dy)
        hit |= (d <= r)
    return hit.astype(np.float32)


def render_diamond(env):
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
    dia = cond["shape"] == "diamond"; rot = cond.get("rot", 0)
    imgs = []
    for s in st:
        env.state = s.copy()
        img = render_diamond(env) if dia else env._render_image()
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


def margin_gp_loss(head, safe, unsafe):
    pos, neg = head(safe), head(unsafe)                      # sign>=0 = safe
    Np = max(pos.shape[0], neg.shape[0])
    def resample(x, n):
        if x.shape[0] >= n: return x[:n]
        idx = torch.randint(0, x.shape[0], (n,), device=x.device); return x[idx]
    sd, ud = resample(safe, Np), resample(unsafe, Np)
    alpha = torch.rand(Np, 1, device=DEV)
    interp = (alpha * sd + (1 - alpha) * ud).requires_grad_(True)
    out = head(interp)
    grads = torch.autograd.grad(out, interp, torch.ones_like(out), create_graph=True)[0]
    gnorm = torch.sqrt((grads ** 2).sum(1) + 1e-12)
    gp = ((gnorm - GRAD_THR) ** 2).mean()
    zs = neg.mean() - pos.mean()
    relu = torch.relu(neg).mean() + torch.relu(-pos).mean()
    return ZS_W * zs + RELU_W * relu + GP_W * gp


def margin_nogp_loss(head, safe, unsafe):  # repo _margin_nogp_loss, gamma_lx margin hinge
    return torch.relu(GAMMA_LX - head(safe)).mean() + torch.relu(GAMMA_LX + head(unsafe)).mean()


def train_margin(Z, y, kind, seed=0):
    torch.manual_seed(seed)
    head = MarginHead().to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    safe = Z[y == 0].to(DEV); unsafe = Z[y == 1].to(DEV)
    lossfn = margin_gp_loss if kind == "gp" else margin_nogp_loss
    for _ in range(STEPS):
        bs = min(256, safe.shape[0], unsafe.shape[0])
        s = safe[torch.randint(0, safe.shape[0], (bs,), device=DEV)]
        u = unsafe[torch.randint(0, unsafe.shape[0], (bs,), device=DEV)]
        loss = lossfn(head, s, u)
        opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()


def score(head, Z, y):  # y: 1=unsafe(inside). safe label for metrics = (y==0)
    with torch.no_grad():
        h = head(Z.to(DEV)).cpu().numpy()
    safe = (y == 0).astype(int)
    acc = float(((h >= 0) == (safe == 1)).mean())
    auc = float(roc_auc_score(safe, h)) if 0 < safe.mean() < 1 else float("nan")
    return acc, auc


def main():
    axis = sys.argv[1] if len(sys.argv) > 1 else "color"
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    og_cond, new_cond, og_tag, new_tag = AXES[axis]
    rng = np.random.default_rng(seed)
    x0, x1, y0, y1 = WORLD_BOUNDS
    st = np.stack([rng.uniform(x0, x1, N), rng.uniform(y0, y1, N),
                   rng.uniform(-np.pi, np.pi, N)], 1).astype(np.float32)
    # exact binary failure label per condition's geometry
    y_og = inside_obstacle(st[:, :2], og_cond["shape"])
    y_new = inside_obstacle(st[:, :2], new_cond["shape"])
    env = DubinsSwmEnv(); tf = get_img_preprocessor(source="pixels", target="pixels", img_size=IMG)
    print(f"axis={axis} seed={seed} OG={og_tag} NEW={new_tag} N={N} "
          f"frac_unsafe og={y_og.mean():.3f} new={y_new.mean():.3f}")

    for ename, ck in ENCODERS.items():
        m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
        Z_og = encode(m, render_cond(env, st, og_cond, tf))
        Z_new = encode(m, render_cond(env, st, new_cond, tf))
        del m; torch.cuda.empty_cache()
        for kind in ("gp", "nogp"):
            h_og = train_margin(Z_og[:NTR], y_og[:NTR], kind, seed=seed)
            h_new = train_margin(Z_new[:NTR], y_new[:NTR], kind, seed=seed)
            oa, ou = score(h_og, Z_og[NTR:], y_og[NTR:])          # in-dist
            za, zu = score(h_og, Z_new[NTR:], y_new[NTR:])         # zero-shot
            na, nu = score(h_new, Z_new[NTR:], y_new[NTR:])        # retrained
            print(f"\n=== jepa_{kind}:{ename}  axis={axis} seed={seed} ===")
            for lbl, a, u in (("M_og on og (in-dist)", oa, ou),
                              ("M_og on new (zero-shot)", za, zu),
                              ("M_new on new (retrain)", na, nu)):
                print(f"{lbl:26s} {a:9.3f} {u:7.3f}")
            print(f"KEY jepa_{kind} {ename} axis={axis} rk=0 seed={seed} "
                  f"ogog_acc={oa:.3f} ogog_auc={ou:.3f} zs_acc={za:.3f} zs_auc={zu:.3f} "
                  f"or_acc={na:.3f} or_auc={nu:.3f}")
    print("\ndone")


if __name__ == "__main__":
    main()
