"""Why did the adversarial safety reg (adv_h, L-inf 8/255) cost Dubins but not Reacher? Hypothesis: on Dubins the safety signal is
in pixel colour (red obstacles), so an L-inf ball reaches it and invariance fights safety encoding; on Reacher safety is arm
geometry (invisible box), which a small intensity perturbation cannot move. Test: how far can the adversary move the margin
function of a model trained WITHOUT the reg? PGD (STEPS sign steps, random start) inside an L-inf ball of radius eps on [0, 1]
pixels, pushing unsafe states toward safe (maximize h) and safe states toward unsafe (minimize h). Reports per (benchmark, WM, eps):
median |dh| / gap (gap = mean h safe - mean h unsafe on clean frames) and the flip rate (sign of h changes) per class.
Real+imag MLP margin functions: Dubins margin_gp_imag_red_s0.pt, Reacher reacher_heads_imag*/<wm>/mlp_s0.pt.
Env: N (per class), STEPS, EPS (comma list, in 1/255)."""
import os, sys
import numpy as np, torch, h5py, torch.nn.functional as F
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import stable_worldmodel as swm
from safe_solver import _load_margin_head

G = os.environ.get
N, STEPS, EPS = int(G("N", "200")), int(G("STEPS", "10")), [float(e) / 255 for e in G("EPS", "8,16").split(",")]
DEV = "cuda"
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=DEV).view(1, 3, 1, 1)
CASES = [  # benchmark, h5, wm checkpoint, margin function, held-out frame filter
    ("dubins", "/data/seongbin/lewm/datasets/dubins_expert.h5", "jacobian_w1_dubins/weights_epoch_50.pt", "/data/seongbin/dreamer/lewm/jacobian_w1_dubins/margin_gp_imag_red_s0.pt"),
    ("dubins", "/data/seongbin/lewm/datasets/dubins_expert.h5", "lewm_dubins_jac_safeadv_linf50/weights_epoch_50.pt", "/data/seongbin/dreamer/lewm/lewm_dubins_jac_safeadv_linf50/margin_gp_imag_red_s0.pt"),
    ("reacher", "/data/seongbin/lewm/datasets/reacher.h5", "rc_jacobian/weights_epoch_10.pt", "/data/seongbin/lewm/reacher_heads_imag/rc_jacobian/mlp_s0.pt"),
    ("reacher", "/data/seongbin/lewm/datasets/reacher.h5", "rc_jac_safeadv_linf/weights_epoch_5.pt", "/data/seongbin/lewm/reacher_heads_imag_ep5/rc_jac_safeadv_linf/mlp_s0.pt"),
]
rng = np.random.default_rng(0)
for bench, h5, ck, hp in CASES:
    with h5py.File(h5, "r") as f:
        y = f["failures"][:].astype(int)
        n = len(y); held = np.arange(int(0.9 * n), n)          # last 10% of frames (held-out end of the dataset for both)
        idx = np.sort(np.r_[rng.choice(held[y[held] == 1], N, replace=False), rng.choice(held[y[held] == 0], N, replace=False)])
        x01 = torch.as_tensor(f["pixels"][idx], device=DEV).permute(0, 3, 1, 2).float() / 255.
    lab = torch.as_tensor(y[idx], device=DEV)
    m = swm.wm.utils.load_pretrained(ck).to(DEV).eval().requires_grad_(False)
    head = _load_margin_head(hp, DEV).eval().requires_grad_(False)
    def h_of(x):   # [0, 1] pixels -> h; Dubins frames are 128 px: normalize then resize to 224 as the training preprocessor
        z = (x - MEAN) / STD
        if z.shape[-1] != 224: z = F.interpolate(z, size=224, mode="bilinear", antialias=True, align_corners=False)
        return head(m.projector(m.encoder(z, interpolate_pos_encoding=True).last_hidden_state[:, 0])).view(-1)
    with torch.no_grad(): h0 = torch.cat([h_of(x01[i:i + 50]) for i in range(0, len(x01), 50)])
    gap = (h0[lab == 0].mean() - h0[lab == 1].mean()).item()
    sgn = torch.where(lab == 1, 1.0, -1.0)                       # unsafe: push h up (toward safe); safe: push h down
    row = []
    for eps in EPS:
        hs = []
        for i in range(0, len(x01), 50):
            x, s_ = x01[i:i + 50], sgn[i:i + 50]
            d = (torch.rand_like(x) * 2 - 1) * eps
            for _ in range(STEPS):
                d.requires_grad_(True)
                from torch.nn.attention import sdpa_kernel, SDPBackend
                with sdpa_kernel(SDPBackend.MATH):
                    g, = torch.autograd.grad((s_ * h_of((x + d).clamp(0, 1))).sum(), d)
                d = (d.detach() + eps / 4 * g.sign()).clamp(-eps, eps)
            with torch.no_grad(): hs.append(h_of((x + d).clamp(0, 1)))
        ha = torch.cat(hs); dh = (ha - h0).abs() / gap
        flip_u = ((h0 < 0) & (ha >= 0))[lab == 1].float().mean().item(); flip_s = ((h0 >= 0) & (ha < 0))[lab == 0].float().mean().item()
        row.append(f"eps={eps * 255:.0f}/255: median|dh|/gap={dh.median().item():.3f} unsafe->safe={flip_u:.2f} safe->unsafe={flip_s:.2f}")
    print(f"ADVSENS {bench} {ck.split('/')[0]} (clean gap {gap:.2f}) | " + " | ".join(row), flush=True)
    del m; torch.cuda.empty_cache()
