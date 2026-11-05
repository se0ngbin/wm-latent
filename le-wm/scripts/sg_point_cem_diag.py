"""Is the Point WM's imagined goal cost predictive? One start (GOAL_MODE/GOAL_D as in sg_point_cem), K action sequences of H steps
executed in the real env from the same start. Spearman between imagined final-latent goal cost, the same cost on the encoded REAL final
frame, and the true final distance to the goal. Also: does the imagined latent move at all (imagined vs encoded start)."""
import os, sys, numpy as np, torch
from scipy.stats import spearmanr
sys.argv = [sys.argv[0]]
os.environ.setdefault("N_EPS", "1"); os.environ.setdefault("NB", "1")
import importlib.util
spec = importlib.util.spec_from_file_location("cem", os.path.join(os.path.dirname(os.path.abspath(__file__)), "sg_point_cem.py"))
src = open(spec.origin).read().split("rows, seed = [], SEED0")[0]   # reuse setup (server, model, prep, action scaling), skip the run
g = {"__file__": spec.origin}; exec(compile(src, spec.origin, "exec"), g)
call, prep, model, A_MU, A_SD = g["call"], g["prep"], g["model"], g["A_MU"], g["A_SD"]
K, H, SEED = int(os.environ.get("K", "16")), int(os.environ.get("H", "25")), int(os.environ.get("DSEED", "1000"))
seed = SEED
while True:
    r = call("reset", [seed])[0]
    if r is not None: break
    seed += 1
img0, gimg, info = r
rng = np.random.default_rng(0)
raw = np.clip(rng.normal(0, 0.7, (K, H, 2)) + rng.uniform(-1, 1, (K, 1, 2)), -1, 1)   # executed actions, per-sequence bias + noise
if os.environ.get("CONST"):   # forward / back / turn +1 / turn -1, held for H steps
    raw = np.repeat(np.array([[1., 0], [-1, 0], [0, 1], [0, -1]])[:, None], H, 1); K = 4
z = ((torch.tensor(raw, dtype=torch.float32) - A_MU) / A_SD)                          # model-space actions
real_final, final_pos = [], []
for k in range(K):
    call("reset", [seed])
    for t in range(H): st = call("step", [raw[k, t].tolist()])[0]
    real_final.append(st[0]); final_pos.append(st[2])
with torch.no_grad():
    D = "cuda"
    enc = lambda ims: model.encode({"pixels": prep(ims).to(D)})["emb"][:, -1]
    zg, z0, zr = enc([gimg]), enc([img0]), enc(real_final)
    info_d = {"pixels": prep([img0]).to(D).unsqueeze(1).expand(1, K, 1, 3, 224, 224), "action": torch.zeros(1, K, 1, 2, device=D)}
    out = model.rollout(info_d, z.unsqueeze(0).to(D))["predicted_emb"][0]           # (K, 1+H... , D)
    zi = out[:, -1]
imag_cost = ((zi - zg) ** 2).sum(-1).cpu().numpy(); real_cost = ((zr - zg) ** 2).sum(-1).cpu().numpy()
dist = np.linalg.norm(np.array(final_pos) - info["goal"], axis=1)
print(f"seed={seed} start->goal={np.linalg.norm(info['goal'] - info['start']):.2f} final dist range {dist.min():.2f}-{dist.max():.2f}")
print(f"spearman imagined-cost vs real-cost {spearmanr(imag_cost, real_cost)[0]:.2f} | imagined-cost vs true dist {spearmanr(imag_cost, dist)[0]:.2f} | real-cost vs true dist {spearmanr(real_cost, dist)[0]:.2f}")
print(f"|z_imag_final - z0| mean {(zi - z0).norm(dim=-1).mean():.3f} vs |z_real_final - z0| mean {(zr - z0).norm(dim=-1).mean():.3f} | spread of imagined finals {zi.std(0).norm():.3f} vs real {zr.std(0).norm():.3f}")
if os.environ.get("CONST"): print("CONST fwd/back/turn+/turn-: imagined cost", imag_cost.round(1), "real cost", real_cost.round(1), "true dist", dist.round(2))
print("action stats used: mu", A_MU.numpy().round(3), "sd", A_SD.numpy().round(3))
g["srv"].stdin.close()
