"""CEM only ranks candidates from the SAME start state. Does the purple recolor preserve that within-state ranking even
though cross-state AUC collapses? Per held-out window: NC random action sequences (CEM-like: N(0, 1) in z-scored action
space), rolled out from the red and from the purple context; score = min_t h(z_hat_t) over the horizon (what the
safety-first cost thresholds) and the hinge sum. Report the mean within-window Spearman(red, purple), plus the mean offset.
Usage: diag_within_state.py tag=ckpt [...]"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np, torch
from scipy.stats import spearmanr
import diag_regrounding as D   # windows, px / px_purple, frames, head loading conventions
import stable_worldmodel as swm
from safe_solver import _MarginHead

NC, NWIN = 32, 120
DEV, H, K = D.DEV, D.H, D.K
g = torch.Generator(device=DEV).manual_seed(0)


@torch.no_grad()
def run(tag, ck):
    m = swm.wm.utils.load_pretrained(f"{ck}/weights_epoch_50.pt").to(DEV).eval()
    head = _MarginHead().to(DEV)
    head.load_state_dict(torch.load(f"/data/seongbin/dreamer/lewm/{ck}/margin_gp_fail_red_s0.pt", map_location=DEV)["margin_gp"])
    head.eval()
    enc = lambda xx: m.projector(m.encoder(xx.to(DEV), interpolate_pos_encoding=True).last_hidden_state[:, 0])
    rho_min, rho_hinge, off = [], [], []
    for w in range(NWIN):
        A0 = torch.from_numpy(D.act[w, :H]).to(DEV)                       # real history actions
        fut = torch.randn(NC, K, A0.shape[-1], generator=g, device=DEV)   # candidate futures
        scores = {}
        for nm, P in (("red", D.px), ("purple", D.px_purple)):
            ctx = enc(D.frames(P[w, :H])).unsqueeze(0).expand(NC, -1, -1).clone()
            emb, aa = ctx, A0.unsqueeze(0).expand(NC, -1, -1)
            hs = []
            for k in range(K):
                p = m.predict(emb[:, -H:], m.action_encoder(aa[:, -H:]))[:, -1:]
                emb = torch.cat([emb, p], 1); aa = torch.cat([aa, fut[:, k:k + 1]], 1)
                hs.append(head(p[:, 0].float()))
            hv = torch.stack(hs, 1)                                        # (NC, K)
            scores[nm] = (hv.min(1).values.cpu().numpy(), torch.relu(-hv).sum(1).cpu().numpy())
        rho_min.append(spearmanr(scores["red"][0], scores["purple"][0]).correlation)
        if np.std(scores["red"][1]) > 0 and np.std(scores["purple"][1]) > 0:
            rho_hinge.append(spearmanr(scores["red"][1], scores["purple"][1]).correlation)
        off.append(float(np.mean(scores["purple"][0] - scores["red"][0])))
    print(f"WITHIN {tag} within-start Spearman(min_t h): {np.nanmean(rho_min):.3f} | hinge: "
          f"{np.nanmean(rho_hinge) if rho_hinge else float('nan'):.3f} (n={len(rho_hinge)}) | mean offset purple-red "
          f"{np.mean(off):+.3f}", flush=True)
    del m; torch.cuda.empty_cache()


for spec in sys.argv[1:]:
    t, c = spec.split("=", 1); run(t, c)
