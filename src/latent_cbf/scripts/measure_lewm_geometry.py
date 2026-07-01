"""Geometry diagnostics for an LE-WM (JEPA) checkpoint.

Two measurements per checkpoint, each averaged over a validation batch:

(1) **Achieved Lipschitz of the optimal margin classifier**
    L_margin = E_x [ ||∇_x  m(x)|| ]  along safety-relevant directions
    where m(x) = margin_head(encoder(x)). Equivalently a finite-difference
    proxy for σ_max(D(m∘φ)). The "floor" interpretation: if the encoder
    collapses safety-relevant directions (σ_min(Dφ) → 0), the head must
    amplify hard for safe/unsafe frames to separate, so this number grows.
    We use the gradient of the *trained* margin head with respect to the
    raw image; the head must already be trained on top of the same WM
    checkpoint (use train_margin_lewm.py first).

(2) **One-step Lipschitz of the latent dynamics**
    L_dyn ≈ max_v ||J_ψ v|| / ||v||
    where ψ(z, a) is the JEPA `predict` over the current history window
    (last context embedding perturbed). Estimated via power iteration on
    Jacobian-vector products. The error-amplification interpretation: if
    L_dyn > 1, encoder errors / off-distribution latents blow up under
    rollout; if L_dyn < 1, they contract.

Outputs JSON with both numbers (mean, p95, max) plus a couple of sanity
diagnostics (pos/neg margin means on validation).
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
import torch

_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parent
_LATENT_CBF = _SCRIPTS.parent
_SRC = _LATENT_CBF.parent
_REPO_ROOT = _SRC.parent
for p in [str(_REPO_ROOT), str(_SRC), str(_LATENT_CBF), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.append(p)

from configs import DreamerConfig  # noqa: E402
from dreamer_offline import make_dataset  # noqa: E402
from dreamerv3_torch import tools as dreamer_tools  # noqa: E402
from latent_cbf.adapters import LEWMWorldModel  # noqa: E402


def _features(wm, batch):
    data = wm.preprocess(batch)
    with torch.no_grad():
        embed = wm.encoder(data)
    return data, embed


def measure_margin_lipschitz(wm, data, head_name, n_samples=64):
    """E_x[ ||∇_x (head(encoder(x)))|| ] over validation pixels.
    Returns dict of mean / p95 / max."""
    pixels = data["pixels"]  # (B, T, C, H, W), normalized
    B, T = pixels.shape[:2]
    flat = pixels.reshape(B * T, *pixels.shape[2:]).detach()
    n = min(n_samples, flat.shape[0])
    flat = flat[:n].clone().requires_grad_(True)

    info = {"pixels": flat.unsqueeze(1)}  # JEPA encode expects (B, T, C, H, W)
    out = wm.jepa.encode(info)
    emb = out["emb"][:, 0]  # (n, D)
    m = wm.heads[head_name](emb).reshape(-1)  # (n,)
    grad = torch.autograd.grad(m.sum(), flat, create_graph=False)[0]
    norms = grad.flatten(1).norm(dim=1).detach().cpu().numpy()
    return {
        "mean": float(norms.mean()),
        "p95": float(np.percentile(norms, 95)),
        "max": float(norms.max()),
        "n": int(n),
    }


def measure_dynamics_lipschitz(wm, embed, n_samples=32, n_power_iter=5):
    """σ_max(∂ψ/∂z_t) of the JEPA predictor wrt the last context embedding,
    estimated via power iteration on JVPs. Returns dict of mean / p95 / max.

    Flash/efficient SDPA backends don't implement double-backward, which
    `torch.autograd.functional.jvp` requires; force the math kernel.
    """
    from torch.nn.attention import sdpa_kernel, SDPBackend

    B, T, D = embed.shape
    if T < wm.history_size + 1:
        return {"mean": float("nan"), "note": "batch too short for history window"}

    H = wm.history_size
    sigmas = []
    n = min(n_samples, B)
    with sdpa_kernel(SDPBackend.MATH):
        for b in range(n):
            z_ctx = embed[b, :H].detach().clone().unsqueeze(0)  # (1, H, D)
            a_ctx = torch.zeros((1, H, 1), device=embed.device, dtype=embed.dtype)
            a_emb = wm.action_encoder(a_ctx)  # (1, H, A_emb)

            def fwd(z_perturb):
                z = z_ctx.clone()
                z[:, -1] = z_perturb
                out = wm.predict(z, a_emb)
                return out[:, -1]

            z0 = z_ctx[:, -1].clone()
            v = torch.randn_like(z0)
            v = v / v.norm()
            for _ in range(n_power_iter):
                jv = torch.autograd.functional.jvp(fwd, (z0,), (v,))[1]
                v = (jv / (jv.norm() + 1e-12)).detach()
            jv = torch.autograd.functional.jvp(fwd, (z0,), (v,))[1]
            sigmas.append(float(jv.norm().item()))

    arr = np.asarray(sigmas)
    return {
        "mean": float(arr.mean()),
        "p95": float(np.percentile(arr, 95)),
        "max": float(arr.max()),
        "n": int(len(arr)),
    }


def main(args):
    config = DreamerConfig()
    config.lewm_ckpt_path = args.ckpt
    config.dataset_path = args.buffer
    config.batch_size = 16
    config.batch_length = 8  # need > history_size

    wm = LEWMWorldModel(config, args.ckpt).to(config.device)
    wm.eval()
    if args.margin_ckpt:
        sd = torch.load(args.margin_ckpt, map_location=config.device)
        wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
        wm.heads["margin_nogp"].load_state_dict(sd["margin_nogp"])

    expert_eps = collections.OrderedDict()
    expert_val_eps = collections.OrderedDict()
    dreamer_tools.fill_offline_dataset(config, expert_eps, expert_val_eps)
    eval_ds = make_dataset(expert_val_eps, config)
    batch = next(eval_ds)
    data, embed = _features(wm, batch)

    out = {
        "run_name": args.run_name,
        "epoch": args.epoch,
        "embed_dim": int(wm.embed_dim),
        "history_size": int(wm.history_size),
    }
    if args.margin_ckpt:
        out["L_margin_gp"] = measure_margin_lipschitz(wm, data, "margin_gp", args.n_samples)
        out["L_margin_nogp"] = measure_margin_lipschitz(wm, data, "margin_nogp", args.n_samples)
    out["L_dynamics"] = measure_dynamics_lipschitz(wm, embed, args.n_samples)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_name", required=True)
    parser.add_argument("--epoch", type=int, required=True)
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--buffer", required=True)
    parser.add_argument("--margin_ckpt", default=None,
                        help="train_margin_lewm.py output; if omitted, L_margin is skipped.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_samples", type=int, default=64)
    main(parser.parse_args())
