"""Train margin_gp / margin_nogp heads on top of a frozen LE-WM (JEPA).

Mirrors the safe/unsafe-batch margin training loop from
``dreamer_offline.py`` (``_margin_gp_step`` / ``_margin_nogp_step``) but
operates on LE-WM embeddings instead of Dreamer RSSM features. The output is a
``{margin_gp, margin_nogp}`` state-dict pair that the LE-WM adapter loads at
CBF train/eval time.

Data format: same Dreamer-style buffer used elsewhere in the repo (a generator
yielding batches with keys ``image``, ``action``, ``is_first``,
``is_terminal``, ``failure``). When sweeping over LE-WM variants, point
``--buffer_path`` at the buffer recorded for the env you want to evaluate on
(e.g. a PushT buffer for the pusht sweep).
"""
from __future__ import annotations

import argparse
import collections
import os
import pathlib
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import trange

# Make sibling modules importable.
_THIS = Path(__file__).resolve()
_SCRIPTS = _THIS.parent
_LATENT_CBF = _SCRIPTS.parent              # src/latent_cbf
_SRC = _LATENT_CBF.parent                  # src
_REPO_ROOT = _SRC.parent                   # latent_cbf repo root
for p in [str(_REPO_ROOT), str(_SRC), str(_LATENT_CBF), str(_SCRIPTS)]:
    if p not in sys.path:
        sys.path.append(p)

from dreamerv3_torch import tools  # noqa: E402
from latent_cbf.adapters import LEWMWorldModel  # noqa: E402
from configs import DreamerConfig  # noqa: E402
from dreamer_offline import make_dataset  # noqa: E402


def _margin_gp_loss(head, safe_feat, unsafe_feat, cfg):
    pos = head(safe_feat)
    neg = head(unsafe_feat)
    gp_loss = torch.zeros((), device=pos.device)
    if pos.numel() > 0 and neg.numel() > 0:
        N = max(pos.numel(), neg.numel())
        pos_data = safe_feat
        neg_data = unsafe_feat
        if N > pos_data.shape[0]:
            r = (N + pos_data.shape[0] - 1) // pos_data.shape[0]
            pos_data = pos_data.repeat((r,) + (1,) * (pos_data.dim() - 1))[
                torch.randperm(pos_data.shape[0] * r, device=pos_data.device)[:N]
            ]
        if N > neg_data.shape[0]:
            r = (N + neg_data.shape[0] - 1) // neg_data.shape[0]
            neg_data = neg_data.repeat((r,) + (1,) * (neg_data.dim() - 1))[
                torch.randperm(neg_data.shape[0] * r, device=neg_data.device)[:N]
            ]
        alpha = torch.rand(pos_data.shape[0], 1, device=pos_data.device)
        interp = (alpha * pos_data + (1 - alpha) * neg_data).requires_grad_(True)
        out = head(interp)
        grads = torch.autograd.grad(
            outputs=out, inputs=interp,
            grad_outputs=torch.ones_like(out),
            create_graph=True, retain_graph=True, only_inputs=True,
        )[0]
        gnorm = torch.sqrt((grads.view(grads.shape[0], -1) ** 2).sum(1) + 1e-12)
        gp_loss = ((gnorm - cfg.gradient_thresh) ** 2).mean()

    pos_mean = pos.mean() if pos.numel() > 0 else torch.zeros((), device=pos.device)
    neg_mean = neg.mean() if neg.numel() > 0 else torch.zeros((), device=pos.device)
    zs_loss = neg_mean - pos_mean
    relu_loss = (
        (torch.relu(neg).mean() if neg.numel() > 0 else 0.0)
        + (torch.relu(-pos).mean() if pos.numel() > 0 else 0.0)
    )
    return cfg.zs_weight * zs_loss + cfg.relu_weight * relu_loss + cfg.gp_weight * gp_loss


def _margin_nogp_loss(head, safe_feat, unsafe_feat, cfg):
    pos = head(safe_feat)
    neg = head(unsafe_feat)
    loss = torch.zeros((), device=pos.device)
    if pos.numel() > 0:
        loss = loss + torch.relu(cfg.gamma_lx - pos).mean()
    if neg.numel() > 0:
        loss = loss + torch.relu(cfg.gamma_lx + neg).mean()
    return loss


def main(config, buffer_path, out_path, steps):
    device = config.device
    wm = LEWMWorldModel(config, config.lewm_ckpt_path or config.lewm_run_name).to(device)
    wm.eval()

    # Build the offline dataset using the same loader Dreamer pretraining uses.
    config.dataset_path = buffer_path
    config.batch_size = 32
    config.batch_length = 4
    expert_eps = collections.OrderedDict()
    expert_val_eps = collections.OrderedDict()
    tools.fill_offline_dataset(config, expert_eps, expert_val_eps)
    dataset = make_dataset(expert_eps, config)

    opt_gp = torch.optim.AdamW(wm.heads["margin_gp"].parameters(), lr=3e-4)
    opt_nogp = torch.optim.AdamW(wm.heads["margin_nogp"].parameters(), lr=3e-4)

    for step in trange(steps, desc="margin train", ncols=0):
        batch = next(dataset)
        data = wm.preprocess(batch)
        with torch.no_grad():
            embed = wm.encoder(data)  # (B, T, D)
            # get_feat == embed for the transformer WM (unchanged), but for the
            # GRU WM it is concat(embed, deter_h) (carried memory), so observe.
            post, _ = wm.dynamics.observe(embed, data["action"], data["is_first"])
            feat = wm.dynamics.get_feat(post)

        failure = data["failure"]
        feat_flat = feat.reshape(-1, feat.shape[-1])
        fail_flat = failure.reshape(-1)
        safe_feat = feat_flat[fail_flat == 0]
        unsafe_feat = feat_flat[fail_flat == 1]
        if safe_feat.numel() == 0 or unsafe_feat.numel() == 0:
            continue

        opt_gp.zero_grad(set_to_none=True)
        loss_gp = _margin_gp_loss(wm.heads["margin_gp"], safe_feat, unsafe_feat, config)
        loss_gp.backward()
        opt_gp.step()

        opt_nogp.zero_grad(set_to_none=True)
        loss_nogp = _margin_nogp_loss(wm.heads["margin_nogp"], safe_feat, unsafe_feat, config)
        loss_nogp.backward()
        opt_nogp.step()

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "margin_gp": wm.heads["margin_gp"].state_dict(),
            "margin_nogp": wm.heads["margin_nogp"].state_dict(),
            "lewm_run_name": config.lewm_run_name,
            "embed_dim": wm.embed_dim,
        },
        out_path,
    )
    print(f"saved margin heads -> {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lewm_run_name", type=str, default="")
    parser.add_argument("--lewm_ckpt_path", type=str, default="")
    parser.add_argument("--buffer_path", type=str, required=True)
    parser.add_argument("--out_path", type=str, required=True)
    parser.add_argument("--steps", type=int, default=5000)
    args = parser.parse_args()

    config = DreamerConfig()
    config.wm_backend = "lewm"
    config.lewm_run_name = args.lewm_run_name
    config.lewm_ckpt_path = args.lewm_ckpt_path
    main(config, args.buffer_path, args.out_path, args.steps)
