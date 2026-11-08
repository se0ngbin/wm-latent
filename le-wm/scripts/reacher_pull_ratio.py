"""Calibrate action_separation (pull) for Reacher: measure what the reg constrains on trained WMs, before choosing target/weight.
Exactly ActionSeparationReg's counterfactual: real training windows (history 3, frameskip 5; action block = 5 z-scored (a1, a2)
steps), swap only the LAST context action block with another window's, r = ||pred(z, a) - pred(z, a')|| / ||a - a'|| (last step).
Reports per WM: median r and its spread (q10/q90, CV), r split by joint speed and by distance to the box (is control authority
state-uniform, which pull assumes?), and the latent scale: median ||z_{t+1} - z_t|| per block and median ||a - a'||.
Env: WMS (wm:ep,...), N."""
import os, sys
import numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sg_point_eval as E

G = os.environ.get
WMS = [w.split(":") for w in G("WMS", "rc_baseline:10,rc_jacobian:10,lewm_reacher:50").split(",")]
N, FS, H = int(G("N", "2000")), 5, 3
rng = np.random.default_rng(0)
with h5py.File("/data/seongbin/lewm/datasets/reacher.h5", "r") as f:
    ep, A, Q, V, y = f["ep_idx"][:], f["action"][:], f["qpos"][:], f["qvel"][:], f["failures"][:].astype(int)
    L = H * FS; t = np.arange(len(ep) - L - 1); ok = (ep[t] < 9000) & (ep[t] == ep[t + L])
    st = np.sort(rng.choice(t[ok], N, replace=False))
    fr = np.stack([st + k * FS for k in range(H + 1)], 1)                       # H context frames + the next one
    u = np.unique(fr.ravel()); px = f["pixels"][u][np.searchsorted(u, fr.ravel())].reshape(N, H + 1, 224, 224, 3)
Af = A[~np.isnan(A).any(1)]; mu, sd = Af.mean(0), Af.std(0)
act = np.stack([((A[s + k * FS: s + (k + 1) * FS] - mu) / sd).reshape(-1) for s in st for k in range(H)]).reshape(N, H, 2 * FS).astype(np.float32)
speed = np.abs(V[fr[:, H - 1]]).sum(1)
cs = np.r_[0, np.cumsum(y)]; nearbox = (cs[np.minimum(fr[:, H - 1] + 26, len(y))] - cs[fr[:, H - 1]]) > 0   # box within 25 steps
MEAN = torch.tensor([0.485, 0.456, 0.406], device=E.DEV).view(1, 3, 1, 1); STD = torch.tensor([0.229, 0.224, 0.225], device=E.DEV).view(1, 3, 1, 1)
for wm, epn in WMS:
    m = E.swm.wm.utils.load_pretrained(f"{wm}/weights_epoch_{epn}.pt").to(E.DEV).eval()
    with torch.no_grad():
        Z = []
        for s in range(0, N, 100):
            x = (torch.as_tensor(px[s:s + 100].reshape(-1, 224, 224, 3), device=E.DEV).permute(0, 3, 1, 2).float() / 255. - MEAN) / STD
            Z.append(m.projector(m.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]).view(-1, H + 1, 192))
        Z = torch.cat(Z); a = torch.from_numpy(act).to(E.DEV)
        perm = torch.from_numpy(rng.permutation(N)).to(E.DEV)
        alt = a.clone(); alt[:, -1] = a[perm, -1]
        dz, da = [], []
        for s in range(0, N, 250):
            sl = slice(s, s + 250); z = Z[sl, :H]
            p = m.predict(z, m.action_encoder(a[sl]))[:, -1]; q = m.predict(z, m.action_encoder(alt[sl]))[:, -1]
            dz.append((p - q).norm(dim=-1)); da.append((a[sl, -1] - alt[sl, -1]).norm(dim=-1))
        dz, da = torch.cat(dz).cpu().numpy(), torch.cat(da).cpu().numpy()
        step = (Z[:, H] - Z[:, H - 1]).norm(dim=-1).median().item()
    r = dz / np.maximum(da, 1e-8); v = da > 1e-4; r = r[v]
    lo, hi = speed[v] < np.median(speed[v]), nearbox[v]
    print(f"PULLRATIO {wm} median_r={np.median(r):.3f} q10={np.quantile(r, .1):.3f} q90={np.quantile(r, .9):.3f} cv={r.std() / r.mean():.2f} | "
          f"slow={np.median(r[lo]):.3f} fast={np.median(r[~lo]):.3f} near_box={np.median(r[hi]):.3f} far={np.median(r[~hi]):.3f} | "
          f"median ||dz_step||={step:.3f} ||da||={np.median(da[v]):.3f} ||dz_swap||={np.median(dz[v]):.3f}", flush=True)
    del m; torch.cuda.empty_cache()
