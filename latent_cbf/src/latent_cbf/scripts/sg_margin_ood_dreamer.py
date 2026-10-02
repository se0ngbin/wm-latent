"""Dreamer margin OOD eval on the safety-gym CarGoal port (matches sg_margin_ood.py for LEWM).
Encodes the paired blue/purple eval set via the sg RSSM (observe->get_feat), trains a probe
margin (same WGAN-GP as the LEWM eval, inlined) on BLUE features, scores zero-shot on purple,
retrain, + cos_la. 2D-action WorldModel build (build_dreamer_wm hardcodes 1D)."""
import os, sys
from pathlib import Path
_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1]), str(_THIS.parent)]:
    if p not in sys.path: sys.path.append(p)
import numpy as np, torch, torch.nn as nn
import gymnasium as gym
from configs import DreamerConfig, Config
from dreamerv3_torch.models import WorldModel
from sklearn.metrics import roc_auc_score

DEV = "cuda:0"
EVAL = "/data/seongbin/lewm/datasets/sg_ood_eval.npz"
RSSM = "/data/seongbin/dreamer/dreamer/sg_baseline/rssm_ckpt.pt"
NTR = 6000
import os as _os
SEED = int(_os.environ.get("SG_SEED","0"))
GRAD_THR, ZS_W, RELU_W, GP_W, GAMMA_LX, LR, STEPS = 0.1, 0.1, 1.0, 10.0, 0.75, 3e-4, 5000

class MarginHead(nn.Module):
    def __init__(self, inp_dim, units=512):
        super().__init__(); self.layers = nn.Sequential(); d = inp_dim
        for i in range(2):
            self.layers.add_module(f"l{i}", nn.Linear(d, units, bias=False))
            self.layers.add_module(f"n{i}", nn.LayerNorm(units, eps=1e-3))
            self.layers.add_module(f"a{i}", nn.SiLU()); d = units
        self.mean_layer = nn.Linear(units, 1)
    def forward(self, x): return self.mean_layer(self.layers(x))

def margin_gp_loss(head, safe, unsafe):
    pos, neg = head(safe), head(unsafe); Np = max(pos.shape[0], neg.shape[0])
    def rs(x, n):
        if x.shape[0] >= n: return x[:n]
        return x[torch.randint(0, x.shape[0], (n,), device=x.device)]
    sd, ud = rs(safe, Np), rs(unsafe, Np)
    a = torch.rand(Np, 1, device=DEV); interp = (a*sd + (1-a)*ud).requires_grad_(True)
    g = torch.autograd.grad(head(interp), interp, torch.ones(Np,1,device=DEV), create_graph=True)[0]
    gp = ((torch.sqrt((g**2).sum(1)+1e-12) - GRAD_THR)**2).mean()
    return ZS_W*(neg.mean()-pos.mean()) + RELU_W*(torch.relu(neg).mean()+torch.relu(-pos).mean()) + GP_W*gp

def margin_nogp_loss(head, safe, unsafe):
    return torch.relu(GAMMA_LX - head(safe)).mean() + torch.relu(GAMMA_LX + head(unsafe)).mean()

def train_margin(Z, y, kind, seed=0):
    torch.manual_seed(seed); head = MarginHead(Z.shape[1]).to(DEV)
    opt = torch.optim.AdamW(head.parameters(), lr=LR)
    safe = Z[y==0].to(DEV); unsafe = Z[y==1].to(DEV); lf = margin_gp_loss if kind=="gp" else margin_nogp_loss
    for _ in range(STEPS):
        bs = min(256, safe.shape[0], unsafe.shape[0])
        s = safe[torch.randint(0, safe.shape[0], (bs,), device=DEV)]
        u = unsafe[torch.randint(0, unsafe.shape[0], (bs,), device=DEV)]
        loss = lf(head, s, u); opt.zero_grad(); loss.backward(); opt.step()
    return head.eval()

def score(head, Z, y):
    with torch.no_grad(): h = head(Z.to(DEV)).cpu().numpy()
    safe = (y==0).astype(int); auc = float(roc_auc_score(safe, h)) if 0<safe.mean()<1 else float("nan")
    return auc

def build_wm2d(rssm_ckpt):
    ec = Config().environment; cfg = DreamerConfig()
    cfg.turnRate = ec.max_angular_velocity
    cfg.x_min, cfg.x_max = ec.world_bounds[0], ec.world_bounds[1]
    cfg.y_min, cfg.y_max = ec.world_bounds[2], ec.world_bounds[3]; cfg.size = [128, 128]
    act = gym.spaces.Box(-cfg.turnRate, cfg.turnRate, (2,), np.float32)
    low = np.array([cfg.x_min, cfg.y_min, -np.pi]); high = np.array([cfg.x_max, cfg.y_max, np.pi])
    obs = gym.spaces.Dict({"state": gym.spaces.Box(low, high, dtype=np.float32),
        "obs_state": gym.spaces.Box(-1, 1, (2,), np.float32),
        "image": gym.spaces.Box(0, 255, (128, 128, 3), np.uint8)})
    cfg.num_actions = 2
    wm = WorldModel(obs, act, 0, cfg).to(DEV).eval()
    ck = torch.load(rssm_ckpt, map_location=DEV)
    sd = {k[14:]: v for k, v in ck["agent_state_dict"].items() if k.startswith("_wm._orig_mod.")}
    wm.load_state_dict(sd); return wm

def feats(wm, imgs, ths, bs=128):
    out = []
    for i in range(0, len(imgs), bs):
        ch, tc = imgs[i:i+bs], ths[i:i+bs]; b = ch.shape[0]
        obs_state = np.stack([np.cos(tc), np.sin(tc)], -1)[:, None].astype(np.float32)
        batch = {"image": ch[:, None], "obs_state": obs_state, "action": np.zeros((b, 1, 2), np.float32),
                 "is_first": np.ones((b, 1), np.float32), "is_terminal": np.zeros((b, 1), np.float32)}
        data = wm.preprocess(batch); embed = wm.encoder(data)
        st, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
        out.append(wm.dynamics.get_feat(st)[:, -1].detach().cpu())
    return torch.cat(out)

def cosla(mu, la): return float(abs((mu@la)/(np.linalg.norm(mu)*np.linalg.norm(la)+1e-12)))

def main():
    d = np.load(EVAL); blue, ood, y = d["blue"], d["ood"], d["failure"].astype(int); th = d["state"][:, 2]
    print(f"eval n={len(y)} unsafe_frac={y.mean():.3f} ood_color={d['ood_color']}")
    wm = build_wm2d(RSSM)
    import os
    AXES = [("color", blue, ood, th, y)]
    cr = "/data/seongbin/lewm/datasets/sg_camrot_eval.npz"   # REAL 90deg camera rotation
    if os.path.exists(cr):
        dr = np.load(cr); AXES.append(("rotate", dr["base"], dr["camrot"], dr["state"][:,2], dr["failure"].astype(int)))
    sc, sb = "/data/seongbin/lewm/datasets/sg_shape_circle.npz", "/data/seongbin/lewm/datasets/sg_shape_box.npz"
    if os.path.exists(sc) and os.path.exists(sb):
        dc, db = np.load(sc), np.load(sb)
        AXES.append(("diamond", dc["img"], db["img"], dc["state"][:,2], dc["failure"].astype(int)))
    for axis, bi, oi, ath, ay in AXES:
        Zb, Zo = feats(wm, bi, ath), feats(wm, oi, ath)
        Zbn, Zon = Zb.numpy(), Zo.numpy()
        mu = (Zon - Zbn).mean(0); la = Zbn[ay==1].mean(0) - Zbn[ay==0].mean(0)
        for kind in ("gp", "nogp"):
            hb = train_margin(Zb[:NTR], ay[:NTR], kind, SEED); ho = train_margin(Zo[:NTR], ay[:NTR], kind, SEED)
            print(f"KEY sg_{kind} dreamer axis={axis} ogog_auc={score(hb,Zb[NTR:],ay[NTR:]):.3f} "
                  f"zs_auc={score(hb,Zo[NTR:],ay[NTR:]):.3f} or_auc={score(ho,Zo[NTR:],ay[NTR:]):.3f} cos_la={cosla(mu,la):.3f}", flush=True)
    print("SG_MARGIN_DREAMER_DONE")

if __name__ == "__main__":
    main()
