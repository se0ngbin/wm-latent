"""CartPole offline dataset in the dubins_expert.h5 schema (+ per-frame `margin` l, safety `value` V, and
`unsafe` = V<0: failed (l<0) or doomed (l>=0, V<0); ~17% failed / ~15% doomed after the filter).
Starts: half near-upright at a random cart pose, half uniform over the not-yet-failed box (incl. doomed
states). Policy: V-greedy, each step replaced by a random action with a per-episode prob eps ~ U(0, eps_max).
Episodes run T steps or until `post` frames after the first failure. All failing episodes are kept; safe-only
episodes are subsampled to `target_unsafe` failure frames (dubins: 17%). Pixels uncompressed (gzip = 30x slower
training reads, see safety-gym port)."""
import os, sys, argparse
import numpy as np, torch, h5py
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_dp import load, step, margin, greedy
from render import render, POLE_PALETTE


def rollout(g, V, k, n, T, post, eps_max, gen):
    dev = V.device
    u = torch.rand(n, 4, device=dev, generator=gen)
    box = g.lo + (g.hi - g.lo) * u
    up = torch.stack([(2 * u[:, 0] - 1) * 2.0, (2 * u[:, 1] - 1) * 1.5, (2 * u[:, 2] - 1) * 0.1, (2 * u[:, 3] - 1) * 0.5], -1)
    s = torch.where((torch.rand(n, 1, device=dev, generator=gen) < 0.5), up, box)
    s = torch.where(margin(s)[:, None] < 0, up, s)          # start not-yet-failed
    eps = torch.rand(n, device=dev, generator=gen) * eps_max
    S, A = [], []
    for _ in range(T):
        a = greedy(g, V, s, k)
        rnd = torch.rand(n, device=dev, generator=gen) < eps
        a = torch.where(rnd, torch.randint(0, 2, (n,), device=dev, generator=gen), a)
        S.append(s); A.append(a); s = step(s, a)
    S, A = torch.stack(S, 1), torch.stack(A, 1)              # (n,T,4), (n,T)
    fail = margin(S) < 0
    first = torch.where(fail.any(1), fail.float().argmax(1), torch.full((n,), T, device=dev))
    L = torch.clamp(first + post, max=T)
    return S.cpu().numpy(), A.cpu().numpy(), L.cpu().numpy(), fail.any(1).cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="/data/seongbin/lewm/datasets/cartpole_value.npz")
    ap.add_argument("--out", default="/data/seongbin/lewm/datasets/cartpole_expert.h5")
    ap.add_argument("--frames", type=int, default=170000)
    ap.add_argument("--target_unsafe", type=float, default=0.17)
    ap.add_argument("--T", type=int, default=64)
    ap.add_argument("--post", type=int, default=8)
    ap.add_argument("--eps_max", type=float, default=0.6)
    ap.add_argument("--pole", default="tan")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    g, V, k = load(a.value)
    gen = torch.Generator(device=V.device).manual_seed(a.seed)
    rng = np.random.default_rng(a.seed)
    fail_eps, safe_eps = [], []
    while True:
        S, A, L, F = rollout(g, V, k, 4096, a.T, a.post, a.eps_max, gen)
        for s, act, l, f in zip(S, A, L, F):
            if l < 8: continue
            (fail_eps if f else safe_eps).append((s[:l], act[:l]))
        unsafe = sum(int((margin(torch.as_tensor(s)) < 0).sum()) for s, _ in fail_eps)
        tot_fail = sum(len(s) for s, _ in fail_eps)
        need_safe = unsafe / a.target_unsafe - tot_fail
        safe_frames = sum(len(s) for s, _ in safe_eps)
        print(f"  fail_eps={len(fail_eps)} unsafe_f={unsafe} safe_eps={len(safe_eps)}", flush=True)
        if unsafe / a.target_unsafe >= a.frames and safe_frames >= need_safe: break
    # keep failing eps until ~frames*target_unsafe unsafe frames, then safe eps to fill
    rng.shuffle(fail_eps); rng.shuffle(safe_eps)
    kept, u = [], 0
    for e in fail_eps:
        if u >= a.frames * a.target_unsafe: break
        kept.append(e); u += int((margin(torch.as_tensor(e[0])) < 0).sum())
    n_tot = u / a.target_unsafe; n = sum(len(e[0]) for e in kept)
    for e in safe_eps:
        if n >= n_tot: break
        kept.append(e); n += len(e[0])
    rng.shuffle(kept)

    STA = np.concatenate([e[0] for e in kept]).astype(np.float32)
    ACT = (2.0 * np.concatenate([e[1] for e in kept]) - 1.0).astype(np.float32)[:, None]   # ±1 (force ±10 N)
    lens = np.array([len(e[0]) for e in kept]); off = np.concatenate([[0], np.cumsum(lens)[:-1]])
    st = torch.as_tensor(STA, device=V.device)
    MAR = margin(st).cpu().numpy().astype(np.float32)
    VAL = torch.cat([g.interp(V, st[i:i + 100000]) for i in range(0, len(st), 100000)]).cpu().numpy().astype(np.float32)
    col = dict(pole=POLE_PALETTE[a.pole])
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with h5py.File(a.out, "w") as f:
        pix = f.create_dataset("pixels", (len(STA), 128, 128, 3), np.uint8)
        for i in range(0, len(STA), 2048):
            pix[i:i + 2048] = np.stack([render(s, col) for s in STA[i:i + 2048]])
        f["action"] = ACT; f["proprio"] = STA; f["state"] = STA
        f["failures"] = (MAR < 0).astype(np.uint8); f["margin"] = MAR; f["value"] = VAL
        f["unsafe"] = ((VAL < 0) | (MAR < 0)).astype(np.uint8)   # failed OR doomed (l >= 0 > V)
        f["episode_idx"] = np.repeat(np.arange(len(kept)), lens); f["step_idx"] = np.concatenate([np.arange(l) for l in lens])
        f["ep_len"] = lens.astype(np.int64); f["ep_offset"] = off.astype(np.int64)
    doomed = (MAR >= 0) & (VAL < 0)
    print(f"WROTE {a.out} frames={len(STA)} eps={len(kept)} failed={np.mean(MAR < 0):.3f} "
          f"doomed(l>=0,V<0)={doomed.mean():.3f} safe(V>=0)={np.mean(VAL >= 0):.3f}")


if __name__ == "__main__":
    main()
