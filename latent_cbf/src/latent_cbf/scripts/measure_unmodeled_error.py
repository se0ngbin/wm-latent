"""Measure the UN-MODELED latent prediction error of a frozen LE-WM.

Per held-out transition (o, a, o') with privileged true state x:
    z      = enc(o)             (last frame of the H-frame context)
    z_next = enc(o')            ground-truth next latent
    z_hat  = predictor(ctx, a)  predicted next latent
    e      = z_next - z_hat     residual (d_z,)
    g      = d z_hat / d a      action-Jacobian (d_z,) via forward-mode AD
                                (d_a = 1 for Dubins); guard ||g|| < 1e-8
    e_par  = (g.e / ||g||^2) g  component an action tweak could absorb
    e_perp = e - e_par          UN-MODELED component (the target)

Reported (median / p90 / p99 over the held-out set):
    ||e||, ||e_par||, ||e_perp||;  ||e_perp||/||e||;  ||e||/RMS(z);
    L_l * ||e_perp|| with L_l = ||d margin_head / dz|| at z, against the
    observed RANGE of margin-head values (the only dimensionally meaningful
    comparison); all stratified by true distance-to-obstacle ell(x).
Dumped: scatter of ||e_perp|| vs ell, conformal (1-delta) quantiles of
||e_perp|| for delta in {0.1, 0.05, 0.01}, JSON of all stats.

NO training, NO ensembling: single frozen checkpoint, one forward(+forward-AD)
pass per transition batch. Action-context convention matches jepa.rollout
(action at position j = action applied at frame j).

Usage (from latent_cbf/):
  STABLEWM_HOME=/data/seongbin/lewm .venv/bin/python \
      src/latent_cbf/scripts/measure_unmodeled_error.py \
      --lewm_ckpt pixel_lipschitz_dubins/weights_epoch_50.pt \
      --margin_ckpt /data/seongbin/dreamer/lewm/pixel_lipschitz_dubins/margin_hj_imag.pt
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD

_THIS = Path(__file__).resolve()
for p in [str(_THIS.parents[3]), str(_THIS.parents[2]), str(_THIS.parents[1])]:
    if p not in sys.path:
        sys.path.append(p)
for _p in ("/home/seongbin/latent", "/home/seongbin/latent/le-wm"):
    if _p not in sys.path:
        sys.path.append(_p)

from configs import get_default_config
from scripts.run_experiment import create_env_from_config

OBSTACLES = [(0.25, 0.65, 0.5), (0.25, -0.65, 0.5)]
ACT_MEAN, ACT_STD = -0.02112957, 1.1531051  # dataset action stats (z-scoring)


def ell(states: np.ndarray) -> np.ndarray:
    """Signed distance to nearest obstacle edge (negative inside)."""
    d = np.full(states.shape[0], np.inf)
    for ox, oy, r in OBSTACLES:
        d = np.minimum(d, np.linalg.norm(states[:, :2] - np.array([ox, oy]), axis=1) - r)
    return d


def collect_episodes(env, n_eps, ep_len, rng):
    """Held-out rollouts (fresh env, OU actions); half spawn near obstacles so
    the small-|ell| bins are populated. Logs frames, raw actions, true states."""
    episodes = []
    for i in range(n_eps):
        env.reset()
        if i % 2 == 0:  # uniform spawn
            s = rng.uniform([-1.4, -1.4, -np.pi], [1.4, 1.4, np.pi])
        else:  # spawn in an annulus around a random obstacle
            ox, oy, r = OBSTACLES[rng.integers(len(OBSTACLES))]
            ang = rng.uniform(-np.pi, np.pi)
            rad = r + rng.uniform(0.02, 0.4)
            s = np.array([ox + rad * np.cos(ang), oy + rad * np.sin(ang),
                          rng.uniform(-np.pi, np.pi)])
            s[:2] = np.clip(s[:2], -1.4, 1.4)
        env.state = s.astype(np.float32)
        frames, states, acts = [env.render()], [env.state.copy()], []
        a = 0.0
        for _ in range(ep_len):
            a = float(np.clip(0.8 * a + rng.normal(0, 0.8), -2.0, 2.0))
            obs, _, term, trunc, _ = env.step(np.array([a], dtype=np.float32))
            frames.append(env.render())
            states.append(env.state.copy())
            acts.append(a)
            if term or trunc:
                break
        if len(acts) >= 4:  # need >= H transitions
            episodes.append((np.stack(frames), np.stack(states), np.array(acts)))
    return episodes


def pct(v, qs=(50, 90, 99)):
    return {f"p{q}": float(np.percentile(v, q)) for q in qs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lewm_ckpt", default="pixel_lipschitz_dubins/weights_epoch_50.pt")
    ap.add_argument("--margin_ckpt", default=None,
                    help="optional; L_l stats are skipped without it")
    ap.add_argument("--frameskip", type=int, default=1,
                    help="model frameskip: one predictor step consumes a block "
                         "of fs raw actions and jumps fs env steps")
    ap.add_argument("--n_eps", type=int, default=150)
    ap.add_argument("--ep_len", type=int, default=40)
    ap.add_argument("--device", default="cuda:1")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_prefix",
                    default=str(_THIS.parents[3] / "results" / "unmodeled_error"))
    args = ap.parse_args()

    dc_mod = __import__("configs", fromlist=["DreamerConfig"])
    dcfg = dc_mod.DreamerConfig()
    dcfg.device = args.device

    from latent_cbf.adapters import LEWMWorldModel

    wm = LEWMWorldModel(dcfg, args.lewm_ckpt).to(args.device).eval()
    head = None
    if args.margin_ckpt:
        sd = torch.load(args.margin_ckpt, map_location=args.device)
        wm.heads["margin_gp"].load_state_dict(sd["margin_gp"])
        head = wm.heads["margin_gp"]
    H = wm.history_size
    fs = args.frameskip
    dev = args.device

    env = create_env_from_config(get_default_config())
    rng = np.random.default_rng(args.seed)
    print("collecting held-out rollouts...")
    episodes = collect_episodes(env, args.n_eps, args.ep_len, rng)
    env.close()
    print(f"{len(episodes)} episodes")

    E, EPAR, EPERP, GNORM, LL, HVAL, DIST, ZNRM = [], [], [], [], [], [], [], []
    n_skipped = 0

    from torch.nn.attention import SDPBackend, sdpa_kernel

    for frames, states, acts in episodes:
        # macro clock: frames at env steps 0, fs, 2fs, ...; action block j =
        # raw actions [j*fs, (j+1)*fs). For fs=1 this is the raw clock.
        n_macros = len(acts) // fs
        if n_macros < H + 1:
            continue
        key_idx = np.arange(n_macros + 1) * fs           # frame indices
        blocks = (acts[: n_macros * fs].reshape(n_macros, fs) - ACT_MEAN) / ACT_STD
        with torch.no_grad():
            data = wm.preprocess(
                {"image": torch.from_numpy(frames[key_idx][None]).float()})
            emb = wm.encoder(data)[0].clone()             # (n_macros+1, D)

        ts = list(range(H - 1, n_macros))
        # context windows (jepa.rollout convention: block j applied at frame j)
        he = torch.stack([emb[t - H + 1 : t + 1] for t in ts])            # (N,H,D)
        aw = torch.tensor(np.stack([blocks[t - H + 1 : t + 1] for t in ts]),
                          device=dev, dtype=torch.float32)                # (N,H,fs)
        z_next = torch.stack([emb[t + 1] for t in ts])                    # (N,D)
        z_cur = he[:, -1]
        N = he.shape[0]

        # z_hat and g = d z_hat / d a_t via forward-mode AD, one pass per
        # action dim (d_a = fs). Efficient-attention kernels lack forward-AD;
        # force the math backend inside the dual level.
        g_cols = []
        for j in range(fs):
            tangent = torch.zeros_like(aw)
            tangent[:, -1, j] = 1.0
            with sdpa_kernel([SDPBackend.MATH]), fwAD.dual_level():
                aw_dual = fwAD.make_dual(aw, tangent)
                act_emb = wm.action_encoder(aw_dual)                      # (N,H,A_emb)
                pred = wm.predict(he, act_emb)[:, -1]                     # (N,D)
                z_hat, gj = fwAD.unpack_dual(pred)
            if gj is None:
                raise RuntimeError("forward-mode AD produced no tangent")
            g_cols.append(gj.detach())
        z_hat = z_hat.detach()
        g = torch.stack(g_cols, dim=-1)                                   # (N,D,fs)

        e = z_next - z_hat                                                # (N,D)
        gfro = g.flatten(1).norm(dim=-1)                                  # (N,)
        ok = gfro > 1e-8
        n_skipped += int((~ok).sum())
        # projector onto im(g): coef = (g^T g)^-1 g^T e (regularized)
        gtg = g.transpose(1, 2) @ g                                       # (N,fs,fs)
        gte = (g.transpose(1, 2) @ e.unsqueeze(-1))                       # (N,fs,1)
        eye = torch.eye(fs, device=dev).expand_as(gtg)
        coef = torch.linalg.solve(gtg + 1e-10 * eye, gte)                 # (N,fs,1)
        e_par = (g @ coef).squeeze(-1)                                    # (N,D)
        e_perp = e - e_par

        if head is not None:
            z_req = z_cur.detach().requires_grad_(True)
            hv = head(z_req).squeeze(-1)
            (gz,) = torch.autograd.grad(hv.sum(), z_req)
        m = ok.cpu().numpy()
        E.append(e.norm(dim=-1).cpu().numpy()[m])
        EPAR.append(e_par.norm(dim=-1).cpu().numpy()[m])
        EPERP.append(e_perp.norm(dim=-1).cpu().numpy()[m])
        GNORM.append(gfro.cpu().numpy()[m])
        if head is not None:
            LL.append(gz.norm(dim=-1).detach().cpu().numpy()[m])
            HVAL.append(hv.detach().cpu().numpy()[m])
        ZNRM.append(z_cur.norm(dim=-1).cpu().numpy()[m])
        DIST.append(ell(states[key_idx[np.array(ts)]])[m])

    E, EPAR, EPERP = map(np.concatenate, (E, EPAR, EPERP))
    GNORM, DIST, ZNRM = map(np.concatenate, (GNORM, DIST, ZNRM))
    n = len(E)
    rms_z = float(np.sqrt(np.mean(ZNRM ** 2)))
    if head is not None:
        LL, HVAL = map(np.concatenate, (LL, HVAL))
        h_range = float(HVAL.max() - HVAL.min())
        ll_eperp = LL * EPERP
    else:
        h_range, ll_eperp = float("nan"), np.full(n, np.nan)

    print(f"\n===== unmodeled latent error: {args.lewm_ckpt} =====")
    print(f"transitions: {n} (skipped ||g||<1e-8: {n_skipped})")
    print(f"RMS(z) = {rms_z:.3f}   d_z = {192}   ||g|| median = {np.median(GNORM):.4f}")
    rep = {
        "ckpt": args.lewm_ckpt, "n": n, "n_skipped": int(n_skipped),
        "rms_z": rms_z, "g_norm": pct(GNORM),
        "e": pct(E), "e_par": pct(EPAR), "e_perp": pct(EPERP),
        "ratio_perp": pct(EPERP / np.maximum(E, 1e-12)),
        "e_over_rmsz": pct(E / rms_z),
        "frameskip": args.frameskip,
    }
    keys = ["e", "e_par", "e_perp", "ratio_perp", "e_over_rmsz"]
    if head is not None:
        rep.update({"L_l": pct(LL), "Ll_eperp": pct(ll_eperp),
                    "margin_range": h_range,
                    "Ll_eperp_over_range": pct(ll_eperp / h_range)})
        keys.append("Ll_eperp")
    for k in keys:
        print(f"{k:>18}: " + "  ".join(f"{q}={v:.4f}" for q, v in rep[k].items()))
    if head is not None:
        print(f"margin-head range = {h_range:.3f}; "
              f"Ll*e_perp / range: " + "  ".join(f"{q}={v:.3f}" for q, v in rep["Ll_eperp_over_range"].items()))

    # conformal quantiles of ||e_perp||
    srt = np.sort(EPERP)
    conf = {}
    for d in (0.1, 0.05, 0.01):
        k = min(math.ceil((n + 1) * (1 - d)), n) - 1
        conf[f"delta={d}"] = float(srt[k])
    rep["conformal_eperp"] = conf
    print("conformal (1-delta) quantiles of ||e_perp||:",
          "  ".join(f"{k}: {v:.4f}" for k, v in conf.items()))

    # stratify by distance-to-obstacle
    bins = [(-np.inf, 0.0), (0.0, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 0.8), (0.8, np.inf)]
    rep["by_distance"] = {}
    print(f"\n{'ell bin':>14} {'n':>5} {'e_perp p50':>11} {'e_perp p90':>11} "
          f"{'Ll*ep p50':>10} {'Ll*ep p90':>10} {'/range p90':>10}")
    for lo, hi in bins:
        m = (DIST >= lo) & (DIST < hi)
        key = f"[{lo:.1f},{hi:.1f})"
        if m.sum() < 5:
            print(f"{key:>14} {int(m.sum()):>5}  (too few)")
            continue
        row = {"n": int(m.sum()), "e_perp": pct(EPERP[m])}
        if head is not None:
            row["Ll_eperp"] = pct(ll_eperp[m])
            row["Ll_eperp_over_range_p90"] = float(
                np.percentile(ll_eperp[m], 90) / h_range)
        rep["by_distance"][key] = row
        ll50 = row.get("Ll_eperp", {}).get("p50", float("nan"))
        ll90 = row.get("Ll_eperp", {}).get("p90", float("nan"))
        llr = row.get("Ll_eperp_over_range_p90", float("nan"))
        print(f"{key:>14} {row['n']:>5} {row['e_perp']['p50']:>11.4f} "
              f"{row['e_perp']['p90']:>11.4f} {ll50:>10.4f} {ll90:>10.4f} {llr:>10.3f}")

    # dumps
    tag = Path(args.lewm_ckpt).parts[0]
    out_json = f"{args.out_prefix}_{tag}.json"
    Path(out_json).parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(rep, f, indent=2)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(DIST, EPERP, s=4, alpha=0.3)
    ax.set_xlabel("true distance to obstacle edge  ell(x)")
    ax.set_ylabel("||e_perp||  (unmodeled latent error)")
    ax.set_title(f"unmodeled error vs obstacle distance — {tag}")
    ax.axvline(0.0, color="r", lw=0.8, ls="--")
    fig.tight_layout()
    out_png = f"{args.out_prefix}_{tag}_scatter.png"
    fig.savefig(out_png, dpi=130)
    print(f"\nsaved {out_json}\nsaved {out_png}")


if __name__ == "__main__":
    main()
