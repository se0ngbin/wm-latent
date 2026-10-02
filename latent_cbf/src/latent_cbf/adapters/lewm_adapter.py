"""Adapter exposing the dreamerv3-torch WorldModel surface on top of a frozen
LE-WM (JEPA) checkpoint produced by le-wm/scripts/run.sh.

The latent_cbf CBF train/eval pipeline calls a narrow interface on the WM:
- preprocess(obs) -> dict (normalizes images, adds 'cont' channel)
- encoder(data)   -> embed (B, T, D)
- dynamics.observe(embed, action, is_first, state=None) -> (post, prior)
- dynamics.imagine_with_action(action, state) -> prior
- dynamics.get_feat(state) -> feat (B[, T], D)
- heads["margin_gp"](feat), heads["margin_nogp"](feat) -> scalar tensor
- heads["cont"](feat).mean -> tensor

LE-WM has a single deterministic embedding (no stoch/deter split). The adapter
returns state dicts with `deter` (the LE-WM embedding) plus a rolling history
window (`hist_emb`, `hist_act`) needed to drive the JEPA autoregressive
predictor inside `imagine_with_action`. Time-aligned shapes are kept so the
existing `state[k][:, -1]` / `state[k][:, [-1]]` slicing idioms remain valid.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

# Make dreamerv3_torch importable (latent_cbf scripts add the parent dir to path
# at runtime; do the same here so this module is importable standalone).
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.append(str(_REPO_ROOT))
from dreamerv3_torch import networks as dreamer_networks  # noqa: E402


class _EncoderProxy:
    """Callable proxy so `wm.encoder(data)` and `wm.encoder.to(dev)` both work
    without re-registering the JEPA parameters under a second module."""

    def __init__(self, jepa):
        self.jepa = jepa

    def __call__(self, data):
        info = self.jepa.encode({"pixels": data["pixels"]})
        return info["emb"]

    def to(self, device):
        return self  # parent already moved


class _ConstHead(nn.Module):
    """Stand-in for the Dreamer cont head: returns an object exposing `.mean`.
    LE-WM has no continuation predictor; latent_cbf only reads `.mean` for
    diagnostics so a constant suffices.
    """

    class _Out:
        def __init__(self, mean):
            self.mean = mean

    def __init__(self, value: float = 1.0):
        super().__init__()
        self.value = value

    def forward(self, feat):
        mean = torch.full(
            feat.shape[:-1] + (1,), self.value, device=feat.device, dtype=feat.dtype
        )
        return _ConstHead._Out(mean)


class _Dynamics:
    """Mimics the RSSM API on top of LE-WM JEPA."""

    def __init__(self, parent: "LEWMWorldModel"):
        self._parent = parent

    @property
    def H(self) -> int:
        return self._parent.history_size

    @property
    def D(self) -> int:
        return self._parent.embed_dim

    def get_feat(self, state):
        if self._parent._is_gru:
            # Dreamer-style feature: current embed + recurrent memory. Some static
            # eval paths build a fake single-frame state without deter_h; there is
            # no history there, so the memory is zeros (get_feat = [embed, 0]).
            deter = state["deter"]
            h = state.get("deter_h")
            if h is None:
                h = deter.new_zeros(*deter.shape[:-1], self._parent.jepa.predictor.deter_dim)
            return torch.cat([deter, h], dim=-1)
        return state["deter"]

    def observe(self, embed, action, is_first, state=None):
        """
        embed:    (B, T, D)
        action:   (B, T, A)
        is_first: (B, T)
        Returns (post, prior) — for LE-WM there is no separate posterior, so
        post == prior.
        """
        if self._parent._is_gru:
            # Carried GRU state: deter_h[t] = h_t summarizing frames strictly
            # BEFORE z_t (the recurrent "prior"), so get_feat(t)=concat(z_t,h_t)
            # mirrors Dreamer's [stoch_t, deter_t]. Seed state for imagine =
            # (z_{T-1}, h_{T-1}).
            predm = self._parent.jepa.predictor
            act_emb = self._parent.action_encoder(action)      # (B, T, A_emb)
            B, T = embed.shape[0], embed.shape[1]
            h = embed.new_zeros(B, predm.deter_dim)
            hs_pre = []
            for t in range(T):
                hs_pre.append(h)
                _, h = predm.step(embed[:, t], act_emb[:, t], h)
            deter_h = torch.stack(hs_pre, 1)                    # (B, T, deter_dim)
            stoch = embed.new_zeros(B, T, 1)
            state_out = {"deter": embed, "deter_h": deter_h, "stoch": stoch}
            return state_out, state_out
        B, T, D = embed.shape
        H = self.H
        device = embed.device

        # LE-WM action encoder: (B, T, A) -> (B, T, A_emb).
        act_emb = self._parent.action_encoder(action)
        A_emb = act_emb.shape[-1]

        # Rolling history at each time step (zero-padded before t=0):
        # hist_emb[:, t] = pad_emb[:, t : t + H]; same for hist_act.
        pad_emb = torch.cat(
            [torch.zeros(B, H - 1, D, device=device, dtype=embed.dtype), embed], dim=1
        )
        pad_act = torch.cat(
            [
                torch.zeros(B, H - 1, A_emb, device=device, dtype=act_emb.dtype),
                act_emb,
            ],
            dim=1,
        )
        hist_emb = pad_emb.unfold(1, H, 1).permute(0, 1, 3, 2).contiguous()  # (B, T, H, D)
        hist_act = pad_act.unfold(1, H, 1).permute(0, 1, 3, 2).contiguous()  # (B, T, H, A_emb)

        stoch = torch.zeros(B, T, 1, device=device, dtype=embed.dtype)
        state_out = {
            "deter": embed,
            "stoch": stoch,
            "hist_emb": hist_emb,
            "hist_act": hist_act,
        }
        return state_out, state_out

    def imagine_with_action(self, action, state):
        """
        action: (B, T_act, A)
        state:  dict whose tensors have the time dim already stripped
                (deter (B, D), stoch (B, 1), hist_emb (B, H, D), hist_act (B, H, A_emb)).
        Returns the rolled-out prior dict with the time dim back in.
        """
        if self._parent._is_gru:
            # Carry the GRU hidden state across imagined steps (unbounded memory,
            # unlike the transformer window). state has deter (B,D) = z_cur and
            # deter_h (B,deter_dim) = memory strictly before z_cur.
            predm = self._parent.jepa.predictor
            proj = self._parent.jepa.pred_proj
            z = state["deter"]
            h = state["deter_h"]
            deters, deter_hs = [], []
            for t in range(action.shape[1]):
                a_emb = self._parent.action_encoder(action[:, t : t + 1])[:, 0]
                z_raw, h = predm.step(z, a_emb, h)
                z = proj(z_raw)
                deters.append(z.unsqueeze(1))
                deter_hs.append(h.unsqueeze(1))
            deter = torch.cat(deters, dim=1)
            deter_h = torch.cat(deter_hs, dim=1)
            stoch = deter.new_zeros(deter.shape[0], deter.shape[1], 1)
            return {"deter": deter, "deter_h": deter_h, "stoch": stoch}
        B, T_act, A = action.shape
        H = self.H
        D = self.D
        device = action.device

        cur_hist_emb = state["hist_emb"]  # (B, H, D)
        cur_hist_act = state["hist_act"]  # (B, H, A_emb)

        deters, hist_embs, hist_acts = [], [], []
        for t in range(T_act):
            a_t = action[:, t : t + 1]  # (B, 1, A)
            a_emb_t = self._parent.action_encoder(a_t)  # (B, 1, A_emb)
            # Roll action history left, append the new action.
            new_hist_act = torch.cat([cur_hist_act[:, 1:], a_emb_t], dim=1)
            # Predict next embedding from current emb history conditioned on
            # the updated action history. The predictor returns predictions for
            # every position; the last one corresponds to the next frame.
            pred = self._parent.predict(cur_hist_emb, new_hist_act)  # (B, H, D)
            next_emb = pred[:, -1]  # (B, D)
            new_hist_emb = torch.cat(
                [cur_hist_emb[:, 1:], next_emb.unsqueeze(1)], dim=1
            )
            deters.append(next_emb.unsqueeze(1))
            hist_embs.append(new_hist_emb.unsqueeze(1))
            hist_acts.append(new_hist_act.unsqueeze(1))
            cur_hist_emb = new_hist_emb
            cur_hist_act = new_hist_act

        deter = torch.cat(deters, dim=1)  # (B, T_act, D)
        stoch = torch.zeros(B, T_act, 1, device=device, dtype=deter.dtype)
        hist_emb = torch.cat(hist_embs, dim=1)
        hist_act = torch.cat(hist_acts, dim=1)
        return {
            "deter": deter,
            "stoch": stoch,
            "hist_emb": hist_emb,
            "hist_act": hist_act,
        }


class LEWMWorldModel(nn.Module):
    """Drop-in replacement for ``dreamerv3_torch.models.WorldModel`` whose
    backbone is a frozen LE-WM JEPA model. Margin heads are fresh MLPs sized
    to LE-WM's embedding dim — they are trained downstream by
    ``scripts/train_margin_lewm.py`` (mirrors the margin training loop in
    ``scripts/dreamer_offline.py``).
    """

    def __init__(self, config, lewm_ckpt):
        super().__init__()
        self._config = config
        self.device = config.device
        # Image size matches LE-WM training (default 224, ViT patch 14).
        self.img_size = int(getattr(config, "lewm_img_size", 224))

        # Load the JEPA model. ``lewm_ckpt`` may be either:
        #   (a) a run-name (resolved under <STABLEWM_HOME>/checkpoints/<run>/), or
        #   (b) a full path to a .pt file with a sibling config.json.
        from stable_worldmodel.wm.utils import load_pretrained

        cache_dir = getattr(config, "lewm_cache_dir", None)
        model = load_pretrained(str(lewm_ckpt), cache_dir=cache_dir)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        self.jepa = model

        # Detect the Dreamer-style recurrent predictor (JEPA+GRU ablation): it
        # carries a GRU hidden state (deter_h) that the adapter threads through
        # observe/imagine and exposes in get_feat = concat(embed, deter_h). For
        # the transformer ARPredictor this stays False and every branch below is
        # a no-op (path byte-identical to before).
        _pred = getattr(self.jepa, "predictor", None)
        self._is_gru = hasattr(_pred, "deter_dim") and hasattr(_pred, "step")

        # Architectural constants pulled from the loaded model.
        self.embed_dim = self._infer_embed_dim()
        self.history_size = self._infer_history_size()

        # ImageNet normalization stats (mirrors le-wm/get_img_preprocessor).
        try:
            from stable_pretraining.data.dataset_stats import ImageNet as _IN
            mean = torch.tensor(_IN["mean"], dtype=torch.float32).view(1, 3, 1, 1)
            std = torch.tensor(_IN["std"], dtype=torch.float32).view(1, 3, 1, 1)
        except Exception:
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("_img_mean", mean)
        self.register_buffer("_img_std", std)

        # Wire dynamics + heads.
        self.dynamics = _Dynamics(self)
        D = self.embed_dim
        units = config.units
        layers = config.margin_head["layers"]
        self.heads = nn.ModuleDict(
            {
                "margin_nogp": dreamer_networks.MLP(
                    D, (), layers, units, config.act, config.norm,
                    device=config.device, return_dist=False, name="Margin NoGP",
                ),
                "margin_gp": dreamer_networks.MLP(
                    D, (), layers, units, config.act, config.norm,
                    device=config.device, return_dist=False, name="Margin GP",
                ),
            }
        )
        # LE-WM has no continuation predictor; latent_cbf only reads
        # heads["cont"].mean for diagnostics, so a constant stub suffices.
        self.heads["cont"] = _ConstHead(value=1.0)

        # `encoder` is read as an attribute by latent_cbf envs and may be
        # `.to(...)`'d; expose a callable proxy rather than rebinding to a
        # method (which has no `.to`).
        self.encoder = _EncoderProxy(self.jepa)

    # ---- introspection ----
    def _infer_embed_dim(self) -> int:
        # JEPA+GRU: the feature the heads consume is concat(embed, deter_h).
        if getattr(self, "_is_gru", False):
            p = self.jepa.predictor
            return int(p.input_dim + p.deter_dim)
        # ARPredictor positional embedding: (1, num_frames, input_dim).
        pos = getattr(self.jepa.predictor, "pos_embedding", None)
        if pos is not None:
            return int(pos.shape[-1])
        # Fallback: probe with a tiny dummy.
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 3, self.img_size, self.img_size)
            out = self.jepa.encode({"pixels": dummy})
            return int(out["emb"].shape[-1])

    def _infer_history_size(self) -> int:
        pos = getattr(self.jepa.predictor, "pos_embedding", None)
        if pos is not None:
            return int(pos.shape[1])
        return 3

    # ---- dreamer-WM API ----
    def preprocess(self, obs):
        """Normalize images and add a 'cont' channel. Mirrors
        ``dreamerv3_torch.models.WorldModel.preprocess`` but produces a
        'pixels' tensor in (B, T, C, H, W) ImageNet-normalized form for the
        ViT-based JEPA encoder.
        """
        out = {}
        for k, v in obs.items():
            if torch.is_tensor(v):
                out[k] = v.to(self.device)
            else:
                dtype = torch.uint8 if k == "image" else torch.float32
                out[k] = torch.tensor(v, device=self.device, dtype=dtype)

        img = out["image"]
        if img.ndim != 5:
            raise ValueError(
                f"LE-WM adapter expects image of shape (B, T, H, W, C); got {tuple(img.shape)}"
            )
        if img.dtype != torch.float32:
            img = img.float()
        B, T = img.shape[:2]
        img = img.permute(0, 1, 4, 2, 3).contiguous()        # (B, T, 3, H, W)
        img = img.view(B * T, *img.shape[2:])
        img = F.interpolate(
            img, size=(self.img_size, self.img_size),
            mode="bilinear", align_corners=False,
        )
        img = img / 255.0
        img = (img - self._img_mean) / self._img_std
        img = img.view(B, T, *img.shape[1:])
        out["pixels"] = img

        if "is_terminal" in out:
            out["cont"] = (1.0 - out["is_terminal"].float()).unsqueeze(-1)
        return out

    def action_encoder(self, action):
        """Embed actions; LE-WM trained with frameskip=5 macros (10D = 5x2D).
        If the caller hands us per-step 2D actions, tile them to match the
        macro shape the underlying Embedder expects. Detect the expected
        input_dim from the Embedder's first conv layer.
        """
        action = action.float()
        expected = int(self.jepa.action_encoder.patch_embed.in_channels)
        cur = int(action.shape[-1])
        if cur != expected:
            if expected % cur != 0:
                raise ValueError(
                    f"action_encoder expects in_channels={expected}, got action "
                    f"with last dim {cur} (not a divisor)."
                )
            rep = expected // cur
            action = action.repeat(*([1] * (action.dim() - 1)), rep)
        return self.jepa.action_encoder(action)

    def predict(self, emb_ctx, act_ctx):
        return self.jepa.predict(emb_ctx, act_ctx)
