import argparse
import functools
import os
import pathlib
import sys
import numpy as np
import ruamel.yaml as yaml


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
sys.path.append(str(pathlib.Path(__file__).parent))

from dreamerv3_torch import models
from dreamerv3_torch import tools

import torch
from torch import nn
from torch import distributions as torchd
import collections

from tqdm import trange
from termcolor import cprint
import matplotlib.pyplot as plt
import gym
from io import BytesIO
from PIL import Image
import matplotlib.patches as patches
import io
to_np = lambda x: x.detach().cpu().numpy()


sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import our modules
from configs import DreamerConfig, Config
from configs.paths import DREAMER_DIR

class Dreamer(nn.Module):
    def __init__(self, obs_space, act_space, config, logger, dataset):
        super(Dreamer, self).__init__()
        self._config = config
        self._logger = logger
        self._should_log = tools.Every(config.log_every)
        batch_steps = config.batch_size * config.batch_length
        self._should_pretrain = tools.Once()
        self._metrics = {}
        # this is update step
        self._step = logger.step // config.action_repeat
        self._update_count = 0
        self._dataset = dataset
        self._wm = models.WorldModel(obs_space, act_space, self._step, config)
        if (
            config.compile and os.name != "nt"
        ):  # compilation is not supported on windows
            self._wm = torch.compile(self._wm)
        self._make_pretrain_opt()
        self._sa_state = {"auc": 0.0, "gap": 0.0, "n": 0, "open": -1, "ema": 0.99}


    def _make_pretrain_opt(self):
        config = self._config
        use_amp = True if config.precision == 16 else False
        if (
            config.steps > 0
            or config.from_ckpt is not None
        ):
            # have separate lrs/eps/clips for actor and model
            # https://pytorch.org/docs/master/optim.html#per-parameter-options
            standard_kwargs = {
                "lr": config.model_lr,
                "eps": config.opt_eps,
                "clip": config.grad_clip,
                "wd": config.weight_decay,
                "opt": config.opt,
                "use_amp": use_amp,
            }
            model_params = {
                "params": list(self._wm.encoder.parameters())
                + list(self._wm.dynamics.parameters())
            }
            model_params["params"] += list(self._wm.heads["decoder"].parameters())
            
            self.pretrain_params = list(model_params["params"]) + list(
            )
            self.pretrain_opt = tools.Optimizer(
                "pretrain_opt", [model_params], **standard_kwargs
            )
            print(
                f"Optimizer pretrain has {sum(param.numel() for param in self.pretrain_params)} variables."
            )

            margin_nogp_params = {
                "params": list(self._wm.heads["margin_nogp"].parameters())
            }
            margin_gp_params = {
                    "params": list(self._wm.heads["margin_gp"].parameters())
                }

            self.margin_nogp_params = list(margin_nogp_params["params"])
            self.margin_gp_params = list(margin_gp_params["params"])

            self.margin_nogp_opt = tools.Optimizer(
                "margin_nogp_opt", [margin_nogp_params], **standard_kwargs)
            self.margin_gp_opt = tools.Optimizer(
                "margin_gp_opt", [margin_gp_params], **standard_kwargs)

            print(
                f"Optimizer margin_nogp has {sum(p.numel() for p in self.margin_nogp_params)} trainable variables."
            )

            print(
                f"Optimizer margin_gp has {sum(p.numel() for p in self.margin_gp_params)} trainable variables."
            )

    def _update_running_metrics(self, metrics):
        for name, value in metrics.items():
            if name not in self._metrics.keys():
                self._metrics[name] = [value]
            else:
                self._metrics[name].append(value)

    def _maybe_log_metrics(self, video_pred_log=False):
        if self._logger is not None:
            logged = False
            if self._should_log(self._step):
                for name, values in self._metrics.items():
                    if not np.isnan(np.mean(values)):
                        self._logger.scalar(name, float(np.mean(values)))
                        self._metrics[name] = []
                logged = True

            if video_pred_log and self._should_log_video(self._step):
                video_pred, video_pred2 = self._wm.video_pred(next(self._dataset))
                self._logger.video("train_openl_agent", to_np(video_pred))
                self._logger.video("train_openl_hand", to_np(video_pred2))
                logged = True

            if logged:
                self._logger.write(fps=True)

    def rssm_step(self, data, step=None, training=True):
        """Unified function for both training and evaluation"""
        wm = self._wm
        data = wm.preprocess(data)
        
        # Train/eval world model and get shared data
        wm_metrics, post, prior = self._world_model_step(data, step, training)
        
        post_detached = {k: v.detach() for k, v in post.items()}
        feat_detached = wm.dynamics.get_feat(post_detached).detach()
        valid = torch.ones_like(data["failure"], dtype=torch.bool)
        if getattr(self._config, "sa_weight", 0.0) > 0.0:
            valid[:, 0] = False   # t=0 label is forced safe by the loader
        safe_data = torch.where((data["failure"] == 0.) & valid)
        unsafe_data = torch.where((data["failure"] == 1.) & valid)
        safe_dataset = feat_detached[safe_data]
        unsafe_dataset = feat_detached[unsafe_data]

        # Train/eval margin heads
        margin_gp_metrics = self._margin_gp_step(safe_dataset, unsafe_dataset, training)
        margin_nogp_metrics = self._margin_nogp_step(safe_dataset, unsafe_dataset, training)
        
        # Combine all metrics
        metrics = {}
        metrics.update(wm_metrics)
        metrics.update(margin_gp_metrics)
        metrics.update(margin_nogp_metrics)
        
        # Add appropriate prefix
        prefix = "model_only_pretrain" if training else "model_only_eval"
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        
        if training:
            self._update_running_metrics(metrics)
            self._maybe_log_metrics()
            self._step += 1
            self._logger.step = self._step
        else:
            return metrics

    def _world_model_step(self, data, step, training=True):
        """Unified world model training/evaluation"""
        metrics = {}
        wm = self._wm
        
        # Choose context manager based on training mode
        grad_context = tools.RequiresGrad(wm) if training else torch.no_grad()
        
        with grad_context:
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                embed = wm.encoder(data)
                post, prior = wm.dynamics.observe(embed, data["action"], data["is_first"])
                
                kl_free = self._config.kl_free
                dyn_scale = self._config.dyn_scale
                rep_scale = self._config.rep_scale
                kl_loss, kl_value, dyn_loss, rep_loss = wm.dynamics.kl_loss(
                    post, prior, kl_free, dyn_scale, rep_scale
                )
                assert kl_loss.shape == embed.shape[:2], kl_loss.shape

                losses = {}
                feat = wm.dynamics.get_feat(post)
                feat_post = feat

                if (step is None or step <= self._config.steps):
                    preds = {}
                    for name, head in wm.heads.items():
                        if "margin" not in name:
                            grad_head = name in self._config.grad_heads
                            feat = wm.dynamics.get_feat(post)
                            feat = feat if grad_head else feat.detach()
                            pred = head(feat)
                            if type(pred) is dict:
                                preds.update(pred)
                            else:
                                preds[name] = pred
                    
                    for name, pred in preds.items():
                        if name == "cont":
                            cont_loss = -pred.log_prob(data[name])
                        elif "margin" not in name:
                            loss = -pred.log_prob(data[name])
                            assert loss.shape == embed.shape[:2], (name, loss.shape)
                            losses[name] = loss
                    recon_loss = sum(losses.values())
                else:
                    recon_loss = torch.tensor(0.0, device=embed.device)
                    cont_loss = torch.tensor(0.0, device=embed.device)
                
                model_loss = kl_loss + recon_loss + cont_loss

                enc_lip_weight = getattr(self._config, "enc_lip_weight", 0.0)
                if enc_lip_weight > 0.0:
                    enc_lip_loss = self._encoder_lipschitz_reg(data, training)
                    model_loss = model_loss + enc_lip_weight * enc_lip_loss
                    metrics["enc_lip_loss"] = to_np(enc_lip_loss)

                if training and getattr(self._config, "sa_weight", 0.0) > 0.0:
                    sa_loss, sa_stats = self._safety_adv_reg(data, feat_post)
                    metrics.update(sa_stats)
                    if sa_loss is not None:
                        model_loss = model_loss + self._config.sa_weight * sa_loss
                        metrics["sa_loss"] = to_np(sa_loss)

                # Only optimize if training
                if training:
                    metrics.update(self.pretrain_opt(torch.mean(model_loss), self.pretrain_params))
        
        # Add metrics
        metrics.update({f"{name}_loss": to_np(loss) for name, loss in losses.items()})
        metrics["kl_loss"] = to_np(kl_loss)
        metrics["dyn_loss"] = to_np(dyn_loss)
        metrics["rep_loss"] = to_np(rep_loss)
        metrics["kl_value"] = to_np(torch.mean(kl_value))
        metrics["cont_loss"] = to_np(cont_loss)
        if not training:
            metrics["model_loss"] = to_np(model_loss)

        with torch.no_grad():
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                metrics["prior_ent"] = to_np(torch.mean(wm.dynamics.get_dist(prior).entropy()))
                metrics["post_ent"] = to_np(torch.mean(wm.dynamics.get_dist(post).entropy()))
        
        return metrics, post, prior

    # ---------------- safety-aware adversarial invariance (port of le-wm SafetyAdvInvarianceReg) --------
    def _safety_adv_reg(self, data, feat_main):
        """Adversarial L_inf pixel invariance of the safety readout (target="head": the in-training
        margin_gp head, frozen params) or of the whole RSSM feature (target="latent", "no h").

        encoder term = posterior feat f(post_t), predictor term = prior feat f(prior_t) (the one-step
        RSSM prediction from the perturbed history), both vs the clean pass, in units of the live
        safe/unsafe separation (head gap, or squared centroid distance for "latent"), with gradient.
        delta: one L_inf field per sequence shared across time, PGD (random start, sign steps,
        per-sample best iterate). Clean and perturbed passes use the SAME RNG seed so the RSSM's
        stochastic samples cancel. t=0 is excluded from label statistics (loader forces it safe).
        Gate: step >= min_steps and EMA head AUC >= auc_gate, then linear ramp. No BatchNorm in the
        dreamer WM, so no clean-stat handling is needed."""
        c, wm, dev = self._config, self._wm, feat_main.device
        st = self._sa_state
        head = wm.heads["margin_gp"]
        params = {k: v.detach() for k, v in head.named_parameters()}
        h = lambda f: torch.func.functional_call(head, params, (f,)).squeeze(-1)
        fail = data["failure"][:, 1:].reshape(-1)
        fm = feat_main[:, 1:].reshape(-1, feat_main.shape[-1])
        # gate statistics (detached)
        with torch.no_grad():
            hm = h(fm.detach())
            if (fail == 0).any() and (fail == 1).any():
                ps, ns = hm[fail == 0], hm[fail == 1]
                auc = (ps[:, None] > ns[None, :]).float().mean()
                st["auc"] = st["ema"] * st["auc"] + (1 - st["ema"]) * float(auc)
                st["gap"] = st["ema"] * st["gap"] + (1 - st["ema"]) * max(float(ps.mean() - ns.mean()), 1e-3)
                st["n"] += 1
        deb = lambda v: v / max(1e-8, 1 - st["ema"] ** st["n"]) if st["n"] > 0 else v
        auc_d, gap_d = deb(st["auc"]), max(deb(st["gap"]), 1e-3)
        if st["open"] < 0 and self._step >= c.sa_min_steps and auc_d >= c.sa_auc_gate:
            st["open"] = self._step
        ramp = 0.0 if st["open"] < 0 else min(1.0, (self._step - st["open"]) / max(1, c.sa_ramp_steps))
        stats = {"sa_auc": auc_d, "sa_gap": gap_d, "sa_ramp": ramp}
        if ramp == 0.0 or not ((fail == 0).any() and (fail == 1).any()):
            return None, stats
        # live separation, with grad
        if c.sa_target == "head":
            hl = h(fm)
            scale = (hl[fail == 0].mean() - hl[fail == 1].mean()).clamp_min(0.1 * gap_d)
        else:
            scale = (fm[fail == 0].mean(0) - fm[fail == 1].mean(0)).pow(2).sum().clamp_min(1e-3)
        n = min(c.sa_n_sub, data["image"].shape[0])
        sub = {k: v[:n] for k, v in data.items() if torch.is_tensor(v)}
        x01 = sub["image"]                                   # (n,T,H,W,C) in [0,1]
        readout = h if c.sa_target == "head" else (lambda f: f)
        seed = int(torch.randint(0, 2 ** 31 - 1, (1,)))

        def feats(img):
            d2 = dict(sub); d2["image"] = img
            with torch.random.fork_rng(devices=[dev]):
                torch.manual_seed(seed)
                post, prior = wm.dynamics.observe(wm.encoder(d2), sub["action"], sub["is_first"])
            return wm.dynamics.get_feat(post), wm.dynamics.get_feat(prior)

        def dev_(r, rc, s):
            return ((r - rc) / s).pow(2) if c.sa_target == "head" else (r - rc).pow(2).sum(-1) / s

        with torch.no_grad():
            fp_c, fq_c = feats(x01)
            rp_c, rq_c = readout(fp_c), readout(fq_c)

        def obj(P, s):
            fp, fq = feats((x01 + P).clamp(0, 1))
            return (c.sa_enc_weight * dev_(readout(fp), rp_c, s).mean(1)
                    + c.sa_pred_weight * dev_(readout(fq), rq_c, s).mean(1))

        eps = c.sa_eps
        s_pgd = scale.detach()
        if os.environ.get("SA_DEBUG"):
            with torch.no_grad():
                print(f"[safety_adv] zero-delta deviation={float(obj(torch.zeros((n, 1) + tuple(x01.shape[2:]), device=dev), s_pgd).abs().max()):.3g}", flush=True)
        P = (torch.rand((n, 1) + tuple(x01.shape[2:]), device=dev) * 2 - 1) * eps
        best_P, best_d, d0 = P, None, None
        for k in range(c.sa_pgd_steps + 1):
            if k < c.sa_pgd_steps:
                P = P.detach().requires_grad_(True)
                with torch.enable_grad():
                    dk = obj(P, s_pgd)
                    gP = torch.autograd.grad(dk.sum(), P)[0]
            else:
                with torch.no_grad():
                    dk = obj(P, s_pgd)
            dk = dk.detach()
            if best_d is None:
                best_d, d0 = dk, dk
            else:
                better = (dk > best_d).view(-1, *([1] * (P.dim() - 1)))
                best_P = torch.where(better, P.detach(), best_P)
                best_d = torch.maximum(dk, best_d)
            if k < c.sa_pgd_steps:
                P = (P.detach() + c.sa_pgd_step_frac * eps * gP.sign()).clamp(-eps, eps)
        d = obj(best_P.detach(), scale).mean()
        stats.update(sa_adv_gain=float(best_d.mean() / (d0.mean() + 1e-8)), sa_scale=float(scale.detach()))
        if self._step % (1 if os.environ.get("SA_DEBUG") else 500) == 0:
            print(f"[safety_adv] step={self._step} " + " ".join(f"{k}={v:.4g}" for k, v in stats.items())
                  + f" sa_loss={float(d):.4g}", flush=True)
        return ramp * d, stats

    def _encoder_branch_fn(self, encoder, key):
        """Return the sub-map embed_branch = f(input[key]) for a single encoder input.

        Only that input's branch is run, so the resulting Jacobian is exactly the
        block of d embed / d input[key] (the other branch is independent of it).
        Valid while each branch has a single key (mlp_keys / cnn_keys), as configured.
        """
        if key in getattr(encoder, "mlp_shapes", {}):
            return encoder._mlp
        if key in getattr(encoder, "cnn_shapes", {}):
            # encoder._cnn does an in-place `obs -= 0.5`; clone so the leaf input
            # is not mutated (autograd forbids in-place on a leaf requiring grad).
            return lambda t: encoder._cnn(t.clone())
        return None

    def _encoder_lipschitz_reg(self, data, training):
        """Pin the encoder's per-input Jacobian norm toward target_L.

        For each key in enc_lip_keys, push ||d embed / d input[key]||_F toward
        target_L so a latent CBF margin converts to an input-space margin:
            ||x - x'|| >= ||embed(x) - embed(x')|| / L.
        "obs_state" regularizes the mlp branch, "image" the cnn (pixel) branch.

        enc_lip_mode selects how ||J||_F is estimated, both unbiased in expectation:
          "exact": Hutchinson VJP, v ~ N(0,I) so E||J^T v||^2 = ||J||_F^2. Requires a
                   second-order (double-backward) graph -- exact but ~5x costlier on pixels.
          "fd":    finite-difference JVP, u ~ N(0,I) so E||J u||^2 = ||J||_F^2, with
                   J u ~= (embed(x + eps*u) - embed(x)) / eps. Two forward passes, no
                   double-backward -- cheap, but carries an O(eps) curvature bias.
        """
        encoder = self._wm.encoder
        target_L = float(self._config.enc_lip_target_L)
        n_probes = int(getattr(self._config, "enc_lip_probes", 4))
        keys = list(getattr(self._config, "enc_lip_keys", ["obs_state"]))
        mode = str(getattr(self._config, "enc_lip_mode", "exact"))
        eps = float(getattr(self._config, "enc_lip_fd_eps", 0.01))

        total = torch.zeros((), device=data[keys[0]].device)
        n_terms = 0
        for key in keys:
            branch = self._encoder_branch_fn(encoder, key)
            if branch is None or key not in data:
                continue
            # fp32 + math path so the (double-)backward is well-behaved.
            with torch.enable_grad(), torch.amp.autocast("cuda", enabled=False):
                if mode == "invariance":
                    # Minimize ||embed(x+delta) - embed(x)||^2, delta = sigma*N(0,I).
                    # No target_L: a pure positive-only smoothness term (the LE-WM
                    # InvarianceReg analog). Relies on the reconstruction loss to
                    # prevent latent collapse.
                    sigma = float(getattr(self._config, "enc_lip_sigma", 0.1))
                    x = data[key].float()
                    z0 = branch(x)
                    inv = torch.zeros(z0.shape[:-1], device=z0.device)
                    for _ in range(n_probes):
                        delta = torch.randn_like(x) * sigma
                        z1 = branch(x + delta)
                        inv = inv + (z1 - z0).pow(2).sum(-1)
                    total = total + (inv / n_probes).mean()
                    n_terms += 1
                    continue
                if mode == "fd":
                    x = data[key].float()
                    z0 = branch(x)  # (B, T, feat); grads flow into encoder
                    fro2 = torch.zeros(z0.shape[:-1], device=z0.device)
                    for _ in range(n_probes):
                        u = torch.randn_like(x)
                        z1 = branch(x + eps * u)
                        fro2 = fro2 + ((z1 - z0).pow(2).sum(-1)) / (eps * eps)
                else:  # "exact"
                    x = data[key].float().detach().requires_grad_(True)
                    z = branch(x)  # (B, T, feat)
                    fro2 = torch.zeros(z.shape[:-1], device=z.device)
                    for _ in range(n_probes):
                        v = torch.randn_like(z)
                        Jtv = torch.autograd.grad(
                            z, x, grad_outputs=v,
                            create_graph=training, retain_graph=True,
                        )[0]  # J^T v, shape of x
                        fro2 = fro2 + Jtv.reshape(*Jtv.shape[:2], -1).pow(2).sum(-1)
                fro_norm = (fro2 / n_probes + 1e-12).sqrt()  # ~= ||J||_F per (B, T)
            total = total + (fro_norm - target_L).pow(2).mean()
            n_terms += 1
        return total / max(n_terms, 1)

    def _margin_gp_step(self, safe_dataset, unsafe_dataset, training=True):
        """Unified margin GP training/evaluation"""
        metrics = {}
        wm = self._wm
        
        # Choose context manager based on training mode
        grad_context = tools.RequiresGrad(wm.heads["margin_gp"])
        
        with grad_context:
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                pos = wm.heads["margin_gp"](safe_dataset)
                neg = wm.heads["margin_gp"](unsafe_dataset)
                
                N = max(pos.numel(), neg.numel())
                gp_loss = torch.tensor(0., device=pos.device)
                
                if pos.numel() > 0 and neg.numel() > 0:
                    # Handle dataset balancing (same logic for both modes)
                    if N > safe_dataset.shape[0]:
                        repeat_times = (N + safe_dataset.shape[0] - 1) // safe_dataset.shape[0]
                        safe_repeated = safe_dataset.repeat((repeat_times,) + (1,) * (safe_dataset.dim() - 1))
                        indices = torch.randperm(safe_repeated.shape[0], device=safe_dataset.device)[:N]
                        pos_data = safe_repeated[indices]
                    else:
                        pos_data = safe_dataset
                        
                    if N > unsafe_dataset.shape[0]:
                        repeat_times = (N + unsafe_dataset.shape[0] - 1) // unsafe_dataset.shape[0]
                        unsafe_repeated = unsafe_dataset.repeat((repeat_times,) + (1,) * (unsafe_dataset.dim() - 1))
                        indices = torch.randperm(unsafe_repeated.shape[0], device=unsafe_dataset.device)[:N]
                        neg_data = unsafe_repeated[indices]
                    else:
                        neg_data = unsafe_dataset
                    
                    # Gradient penalty computation
                    alpha = torch.rand(pos_data.shape[0], 1, device=pos_data.device)
                    interpolates = alpha * pos_data + (1 - alpha) * neg_data
                    interpolates.requires_grad_(True)
                    disc_interpolates = wm.heads["margin_gp"](interpolates)

                    gradients = torch.autograd.grad(
                        outputs=disc_interpolates,
                        inputs=interpolates,
                        grad_outputs=torch.ones_like(disc_interpolates),
                        create_graph=training,  # Only create graph when training
                        retain_graph=training,  # Only retain graph when training
                        only_inputs=True,
                    )[0]
                    gradients = gradients.view(pos_data.shape[0], -1)
                    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)
                    gp_loss = ((gradients_norm - self._config.gradient_thresh) ** 2).mean()

                pos_mean = pos.mean() if pos.numel() > 0 else 0.0
                neg_mean = neg.mean() if neg.numel() > 0 else 0.0
                zero_sum_loss = neg_mean - pos_mean
                
                # old
                #relu_loss = torch.relu(self._config.gamma_lx + neg_mean) + torch.relu(self._config.gamma_lx - pos_mean)
                # new
                #relu_loss = torch.relu(self._config.gamma_lx + neg).mean() + torch.relu(self._config.gamma_lx - pos).mean()
                # punish neg for being positive, and pos for being negative
                neg_relu = torch.relu(neg).mean() if neg.numel() > 0 else 0.0
                pos_relu = torch.relu(-pos).mean() if pos.numel() > 0 else 0.0
                relu_loss = neg_relu + pos_relu


                loss = self._config.zs_weight * zero_sum_loss 
                loss += self._config.relu_weight * relu_loss 
                loss += self._config.gp_weight * gp_loss
                
                # Only optimize if training
                if training:
                    metrics.update(self.margin_gp_opt(loss, wm.heads["margin_gp"].parameters()))
                
                metrics["margin_gp"] = to_np(loss)
                metrics["sign_loss"] = to_np(relu_loss)
                metrics["zs_loss"] = to_np(zero_sum_loss)
                metrics["gp_loss"] = to_np(gp_loss)
                if not training:
                    metrics["pos_mean"] = to_np(pos_mean)
                    metrics["neg_mean"] = to_np(neg_mean)
        
        return metrics

    def _margin_nogp_step(self, safe_dataset, unsafe_dataset, training=True):
        """Unified margin no-GP training/evaluation"""
        metrics = {}
        wm = self._wm
        
        # Choose context manager based on training mode
        grad_context = tools.RequiresGrad(wm.heads["margin_nogp"]) if training else torch.no_grad()
        
        with grad_context:
            with torch.amp.autocast("cuda", enabled=wm._use_amp):
                pos = wm.heads["margin_nogp"](safe_dataset)
                neg = wm.heads["margin_nogp"](unsafe_dataset)
                gamma = self._config.gamma_lx
                lx_loss = 0.0
                
                if pos.numel() > 0:
                    #lx_loss += torch.relu(self._config.gamma_lx - pos.mean()).mean()
                    lx_loss += torch.relu(self._config.gamma_lx - pos).mean()
                if neg.numel() > 0:
                    #lx_loss += torch.relu(self._config.gamma_lx + neg.mean())
                    lx_loss += torch.relu(self._config.gamma_lx + neg).mean()
                
                # Only optimize if training
                if training:
                    metrics.update(self.margin_nogp_opt(lx_loss, wm.heads["margin_nogp"].parameters()))
                
                metrics["margin_nogp"] = lx_loss.item()
                if not training:
                    metrics["pos_mean_nogp"] = to_np(pos.mean()) if pos.numel() > 0 else 0.0
                    metrics["neg_mean_nogp"] = to_np(neg.mean()) if neg.numel() > 0 else 0.0
        
        return metrics

    # Convenience methods
    def train_rssm(self, data, step=None):
        """Training wrapper"""
        return self.rssm_step(data, step, training=True)

    def eval_rssm(self, data, step=None):
        """Evaluation wrapper"""
        return self.rssm_step(data, step, training=False)
        
def count_steps(folder):
    return sum(int(str(n).split("-")[-1][:-4]) - 1 for n in folder.glob("*.npz"))


def make_dataset(episodes, config):
    generator = tools.sample_episodes(episodes, config.batch_length)
    dataset = tools.from_generator(generator, config.batch_size)
    return dataset


# Non-purple obstacle palette (matches le-wm/utils.ObstacleRecolor); red index 0.
COLOR_AUG_PALETTE = np.array([
    [255, 0, 0], [255, 140, 0], [255, 200, 0], [150, 75, 0],
    [128, 128, 0], [250, 128, 114], [140, 0, 0], [0, 150, 150],
], dtype=np.float32)


def color_aug_stream(dataset, palette=COLOR_AUG_PALETTE):
    """Wrap a batch generator: recolor the red obstacle to a random non-purple
    training color per (B) sequence, so the RSSM sees obstacles in many colors while
    purple stays OOD. Obstacle isolated by redness a=relu(min(R-G,R-B))/255; remap
    out = x + a*(C-RED) (exact for red-on-white edges). image: (B,T,H,W,3) uint8."""
    RED = np.array([255., 0., 0.], np.float32)
    while True:
        batch = next(dataset)
        img = batch["image"]
        is_torch = isinstance(img, torch.Tensor)
        arr = (img.detach().cpu().numpy() if is_torch else np.asarray(img)).astype(np.float32)
        for b in range(arr.shape[0]):
            C = palette[np.random.randint(len(palette))]
            x = arr[b]                                   # (T,H,W,3)
            a = np.clip(np.minimum(x[..., 0] - x[..., 1], x[..., 0] - x[..., 2]), 0, None) / 255.0
            for ci in range(3):
                x[..., ci] = np.clip(x[..., ci] + a * (C[ci] - RED[ci]), 0, 255)
            arr[b] = x
        arr = arr.astype(np.uint8)
        batch["image"] = torch.from_numpy(arr).to(img.device) if is_torch else arr
        yield batch


def main(config):
    tools.set_seed_everywhere(config.seed)
    if config.deterministic_run:
        tools.enable_deterministic_run()
    logdir = pathlib.Path(config.logdir).expanduser()
    config.steps //= config.action_repeat
    config.eval_every //= config.action_repeat
    config.log_every //= config.action_repeat

    print("Logdir", logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    # step in logger is environmental step
    step = 0
    if config.debug:
        logger = tools.DebugLogger(logdir, config.action_repeat * step)
    else:
        logger = tools.Logger(logdir, config.action_repeat * step)

    print("Create envs.")
    
    action_space = gym.spaces.Box(
        low=-config.turnRate, high=config.turnRate,
        shape=(getattr(config, "action_dim", 1),), dtype=np.float32
    )
    bounds = np.array([[config.x_min, config.x_max], [config.y_min, config.y_max], [0, 2 * np.pi]])
    low = bounds[:, 0]
    high = bounds[:, 1]
    midpoint = (low + high) / 2.0
    interval = high - low
    gt_observation_space = gym.spaces.Box(
        np.float32(midpoint - interval/2),
        np.float32(midpoint + interval/2),
    )
    image_size = config.size[0] #128
    image_observation_space = gym.spaces.Box(
        low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
    )

    
    obs_observation_space = gym.spaces.Box(
        low=-1, high=1, shape=(2,), dtype=np.float32
    )
    observation_space = gym.spaces.Dict({
            'state': gt_observation_space,
            'obs_state': obs_observation_space,
            'image': image_observation_space
        })


    print("Action Space", action_space)
    config.num_actions = action_space.n if hasattr(action_space, "n") else action_space.shape[0]

    
    expert_eps = collections.OrderedDict()
    expert_val_eps = collections.OrderedDict()

    print(expert_eps)
    tools.fill_offline_dataset(config, expert_eps, expert_val_eps)
    expert_dataset = make_dataset(expert_eps, config)
    if getattr(config, "color_aug", False):
        expert_dataset = color_aug_stream(expert_dataset)
        print("[dreamer_offline] obstacle color_aug ON (purple held out)")
    eval_dataset = make_dataset(expert_val_eps, config)  # eval stays red (in-dist)

    print("Length of training data:", len(expert_eps))
    print("Length of validation data:", len(expert_val_eps))

    print("Simulate agent.")
    agent = Dreamer(
        observation_space,
        action_space,
        config,
        logger,
        expert_dataset,
    ).to(config.device)
    agent.requires_grad_(requires_grad=False)
    if (logdir / "latest.pt").exists():
        checkpoint = torch.load(logdir / "latest.pt")
        agent.load_state_dict(checkpoint["agent_state_dict"])
        tools.recursively_load_optim_state_dict(agent, checkpoint["optims_state_dict"])
        agent._should_pretrain._once = False

    def evaluate(other_dataset=None, eval_prefix=""):
        agent.eval()
        
        eval_policy = functools.partial(agent, training=False)

        # For Logging (1 episode)
        if config.video_pred_log:
            video_pred = agent._wm.video_pred(next(eval_dataset))
            logger.video("eval_recon/openl_agent", to_np(video_pred))

            if other_dataset:
                video_pred = agent._wm.video_pred(next(other_dataset))
                logger.video("train_recon/openl_agent", to_np(video_pred))

        
        logger.write(step=logger.step)

        agent.train()
    # ==================== Pretrain ====================
    total_train_steps = config.steps 
    print(total_train_steps)
    if total_train_steps > 0:
        
        cprint(
            f"Pretraining for {total_train_steps=}",
            color="cyan",
            attrs=["bold"],
        )
        ckpt_name = "rssm_ckpt" 
        best_pretrain_success = float("inf")
        for step in trange(
            total_train_steps,
            desc="Training the RSSM",
            ncols=0,
            leave=False,
        ):
            if (
                ((step + 1) % config.eval_every) == 0
                or step == 1
            ):
                # Add evaluation metrics logging
                agent.eval()
                eval_data = next(eval_dataset)
                eval_metrics = agent.eval_rssm(eval_data, step)
                
                # Log evaluation metrics
                for key, value in eval_metrics.items():
                    logger.scalar(f"eval/{key}", float(np.mean(value)))
                
                # Reset to training mode
                agent.train()
                
                evaluate(
                    other_dataset=expert_dataset, eval_prefix="pretrain"
                )                
                best_pretrain_success = tools.save_checkpoint(
                    ckpt_name, step, None, best_pretrain_success, agent, logdir
                )

            exp_data = next(expert_dataset)
            agent.train_rssm(exp_data, step)
    

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--relu_weight", type=float, default=1.0)
    parser.add_argument("--gp_weight", type=float, default=10.0)
    parser.add_argument("--zs_weight", type=float, default=0.1)
    parser.add_argument("--enc_lip_weight", type=float, default=0.0)
    parser.add_argument("--enc_lip_target_L", type=float, default=1.0)
    parser.add_argument("--enc_lip_probes", type=int, default=4)
    parser.add_argument("--enc_lip_keys", type=str, nargs="+", default=["obs_state"],
                        help="encoder inputs to regularize: obs_state and/or image")
    parser.add_argument("--enc_lip_mode", type=str, default="exact",
                        choices=["exact", "fd", "invariance"],
                        help="exact/fd = regress ||J||_F toward target_L; "
                             "invariance = minimize ||embed(x+delta)-embed(x)||^2 (no target)")
    parser.add_argument("--enc_lip_fd_eps", type=float, default=0.01,
                        help="finite-difference step for enc_lip_mode=fd")
    parser.add_argument("--enc_lip_sigma", type=float, default=0.1,
                        help="perturbation std for enc_lip_mode=invariance (delta = sigma*N(0,I))")
    parser.add_argument("--sa_weight", type=float, default=0.0, help="safety-adv reg weight (0 = off)")
    parser.add_argument("--sa_target", type=str, default="head", choices=["head", "latent"])
    parser.add_argument("--sa_eps", type=float, default=8 / 255)
    parser.add_argument("--sa_n_sub", type=int, default=8)
    parser.add_argument("--sa_pgd_steps", type=int, default=2)
    parser.add_argument("--sa_pgd_step_frac", type=float, default=0.5)
    parser.add_argument("--sa_enc_weight", type=float, default=1.0)
    parser.add_argument("--sa_pred_weight", type=float, default=1.0)
    parser.add_argument("--sa_min_steps", type=int, default=2000)
    parser.add_argument("--sa_auc_gate", type=float, default=0.95)
    parser.add_argument("--sa_ramp_steps", type=int, default=10000)
    parser.add_argument("--logdir", type=str, default=None)
    parser.add_argument("--steps", type=int, default=None,
                        help="override DreamerConfig.steps (WM pretrain steps)")
    parser.add_argument("--color_aug", action="store_true", default=False,
                        help="recolor the red obstacle to random non-purple colors "
                             "per sequence (purple held out as OOD)")
    parser.add_argument("--dataset_path", type=str, default=None,
                        help="override the offline dataset h5 (trajectory-grouped)")
    parser.add_argument("--batch_length", type=int, default=None,
                        help="override RSSM training sequence length (<= traj length)")
    parser.add_argument("--action_dim", type=int, default=None,
                        help="action dimensionality (1 for dubins, 2 for safety-gym Car)")

    args = parser.parse_args()

    config = DreamerConfig()
    if args.dataset_path is not None: config.dataset_path = args.dataset_path
    if args.batch_length is not None: config.batch_length = args.batch_length
    config.action_dim = args.action_dim if args.action_dim is not None else 1
    if args.steps is not None:
        config.steps = args.steps
    config.relu_weight = args.relu_weight
    config.gp_weight = args.gp_weight
    config.zs_weight = args.zs_weight
    config.enc_lip_weight = args.enc_lip_weight
    config.enc_lip_target_L = args.enc_lip_target_L
    config.enc_lip_probes = args.enc_lip_probes
    config.enc_lip_keys = args.enc_lip_keys
    config.enc_lip_mode = args.enc_lip_mode
    config.enc_lip_fd_eps = args.enc_lip_fd_eps
    config.enc_lip_sigma = args.enc_lip_sigma
    for k, v in vars(args).items():
        if k.startswith("sa_"):
            setattr(config, k, v)
    config.logdir = args.logdir if args.logdir else f"{DREAMER_DIR}"
    config.color_aug = args.color_aug
    env_conf = Config()

    config.turnRate = env_conf.max_angular_velocity
    config.x_min = env_conf.environment.world_bounds[0]
    config.x_max = env_conf.environment.world_bounds[1]
    config.y_min = env_conf.environment.world_bounds[2]
    config.y_max = env_conf.environment.world_bounds[3]
    config.size = env_conf.environment.image_size
    
    
    main(config)
