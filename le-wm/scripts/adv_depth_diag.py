"""Would a multi-step adv_h help? adv_h penalizes the change of h under an adversarial pixel perturbation on the context frames and the
1-step prediction only; CCEM reads h on 5 predicted blocks. Per model and prediction step k (0 = encoded start frame, 1..K predicted
blocks of one random CEM-style plan, 1-frame history as in the CEM):
  adversarial : PGD (STEPS sign steps, random start, L-inf EPS on [0, 1] pixels of the start frame) maximizing (h(zhat_k(x+d)) - h(zhat_k(x)))^2,
                separately for each k; median |dh| / gap, and the flip rate (sign of h changes)
  real shift  : start frame rendered under floor 0.3 and camera shift; median |h(zhat_k(x_shift)) - h(zhat_k(x))| / gap
gap = mean h safe - mean h unsafe on clean real frames. If adv_h is robust at k = 0-1 but not at k = 3-5 (adversarially and under the
real shifts), a multi-step adversarial term has room. Held-out frames (episodes >= 9000), N/2 safe + N/2 unsafe; MLP real+imag heads.
Env: WMS (name:ckpt:headsdir,...), N, EPS (1/255), STEPS."""
import os, sys
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np, torch, h5py, gymnasium as gym
from torch.nn.attention import sdpa_kernel, SDPBackend
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
from safe_solver import _load_margin_head
import reacher_shift as RS

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "jac:rc_jacobian/weights_epoch_10.pt:/data/seongbin/lewm/reacher_heads_imag/rc_jacobian,"
                                      "jac+adv_h:rc_jac_safeadv_linf/weights_epoch_10.pt:/data/seongbin/lewm/reacher_heads_imag/rc_jac_safeadv_linf").split(",")]
N, EPS, STEPS, K, FS, DEV = int(G("N", "200")), float(G("EPS", "8")) / 255, int(G("STEPS", "10")), 5, 5, "cuda"
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    y, ep, Q, V, A = f["failures"][:].astype(int), f["ep_idx"][:], f["qpos"][:], f["qvel"][:], f["action"][:]
ho = np.where(ep >= 9000)[0]
idx = np.sort(np.r_[rng.choice(ho[y[ho] == 1], N // 2, replace=False), rng.choice(ho[y[ho] == 0], N // 2, replace=False)])
lab = torch.as_tensor(y[idx], device=DEV)
Af = A[~np.isnan(A).any(1)]
plan = torch.from_numpy((rng.standard_normal((N, 1, 2 * FS)) + 0.7 * rng.standard_normal((N, K, 2 * FS))).astype(np.float32)).to(DEV)
F0, F1 = np.array([[0.1, 0.2, 0.3], [0.2, 0.3, 0.4]]), np.array([[0.45, 0.3, 0.15], [0.55, 0.38, 0.2]])
def render(cond):
    env = gym.make("swm/ReacherDMControl-v0", task="qpos_match", render_mode="rgb_array").unwrapped
    var = {"floor.color": 0.7 * F0 + 0.3 * F1} if cond == "floor0.3" else {}
    env.reset(seed=0, options={"variation_values": var} if var else None)
    sh = cond if cond in RS.APPEARANCE else ""; RS.apply_physics(env, sh)
    return np.stack([RS.corrupt((env.set_state(Q[i], V[i]), env.render())[1], sh, k) for k, i in enumerate(idx)])
X = {c: torch.as_tensor(render(c), device=DEV).permute(0, 3, 1, 2).float() / 255. for c in ["clean", "floor0.3", "cam_shift"]}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
print(f"N={N} eps={EPS * 255:.0f}/255 pgd_steps={STEPS} | per step k = 0 (start frame) .. {K}", flush=True)
for name, ck, hd in WMS:
    m = swm.wm.utils.load_pretrained(ck).to(DEV).eval().requires_grad_(False)
    head = _load_margin_head(f"{hd}/mlp_s0.pt", DEV).eval().requires_grad_(False)
    def hk(x01, pl):   # (n,) * (K+1): h on the encoded start frame and each predicted block
        emb = m.projector(m.encoder((x01 - MEAN) / STD, interpolate_pos_encoding=True).last_hidden_state[:, 0]).unsqueeze(1)
        for k in range(K):
            act = m.action_encoder(pl[:, :k + 1]); emb = torch.cat([emb, m.predict(emb[:, -3:], act[:, -3:])[:, -1:]], 1)
        return head(emb).view(len(x01), K + 1)
    with torch.no_grad(): h0 = torch.cat([hk(X["clean"][i:i + 50], plan[i:i + 50]) for i in range(0, N, 50)])
    gap = (h0[lab == 0, 0].mean() - h0[lab == 1, 0].mean()).item()
    adv = []
    for k in range(K + 1):
        ha = []
        for i in range(0, N, 25):
            x, pl, hc = X["clean"][i:i + 25], plan[i:i + 25], h0[i:i + 25, k]
            d = (torch.rand_like(x) * 2 - 1) * EPS
            for _ in range(STEPS):
                d.requires_grad_(True)
                with sdpa_kernel(SDPBackend.MATH):
                    g, = torch.autograd.grad((hk((x + d).clamp(0, 1), pl)[:, k] - hc).pow(2).sum(), d)
                d = (d.detach() + EPS / 4 * g.sign()).clamp(-EPS, EPS)
            with torch.no_grad(): ha.append(hk((x + d).clamp(0, 1), pl)[:, k])
        ha = torch.cat(ha); dh = (ha - h0[:, k]).abs() / gap; flip = ((ha >= 0) != (h0[:, k] >= 0)).float().mean().item()
        adv.append(f"{dh.median().item():.3f}/{flip:.2f}")
    real = {}
    for c in ["floor0.3", "cam_shift"]:
        with torch.no_grad(): hs = torch.cat([hk(X[c][i:i + 50], plan[i:i + 50]) for i in range(0, N, 50)])
        real[c] = " ".join(f"{v:.3f}" for v in ((hs - h0).abs() / gap).median(0).values.tolist())
    print(f"DEPTH {name} (gap {gap:.2f}) | adversarial median|dh|/gap / flip, k=0..{K}: {'  '.join(adv)} | "
          f"real floor0.3 |dh|/gap: {real['floor0.3']} | real cam_shift: {real['cam_shift']}", flush=True)
    del m; torch.cuda.empty_cache()
