"""Exact-semantics safety value for gym CartPole-v1 (the env's own Euler step, discrete ±force_mag).

    V(s) = max_policy min_{t>=0} l(s_t),   l(s) = min(1 - |x|/2.4, 1 - |theta|/theta_max)

computed by value iteration on a 4D grid,  V <- min(l, max_a V(f(s, a))),  with multilinear
interpolation (torch, GPU). Safe iff V >= 0. l >= 0 but V < 0 = a doomed state: not yet failed, but
no action sequence avoids failure (the dynamics coupling the position-only margin can't see).
Replaces hj_reachability (CPU-only JAX here); DP on the discrete-time map matches the env exactly
up to grid interpolation, which `--check` measures with rollouts of the greedy policy.
"""
import argparse, math
import numpy as np, torch

X_MAX, TH_MAX = 2.4, 12 * 2 * math.pi / 360
PHYS = dict(gravity=9.8, masscart=1.0, masspole=0.1, length=0.5, force_mag=10.0, tau=0.02)


def step(s, a, p=PHYS):
    """Gym CartPole-v1 euler step, batched. s: (...,4) [x, x_dot, th, th_dot]; a in {0,1}."""
    x, xd, th, thd = s.unbind(-1)
    f = (2 * a - 1) * p["force_mag"]
    mt, pml = p["masscart"] + p["masspole"], p["masspole"] * p["length"]
    c, sn = torch.cos(th), torch.sin(th)
    temp = (f + pml * thd ** 2 * sn) / mt
    thacc = (p["gravity"] * sn - c * temp) / (p["length"] * (4 / 3 - p["masspole"] * c ** 2 / mt))
    xacc = temp - pml * thacc * c / mt
    t = p["tau"]
    return torch.stack([x + t * xd, xd + t * xacc, th + t * thd, thd + t * thacc], -1)


def margin(s):
    return torch.minimum(1 - s[..., 0].abs() / X_MAX, 1 - s[..., 2].abs() / TH_MAX)


class Grid:
    def __init__(self, n, vmax, thdmax, dev):
        self.lo = torch.tensor([-X_MAX, -vmax, -TH_MAX, -thdmax], device=dev)
        self.hi = -self.lo
        self.n = n
        axes = [torch.linspace(l, h, n, device=dev) for l, h in zip(self.lo.tolist(), self.hi.tolist())]
        self.axes = axes
        self.pts = torch.stack(torch.meshgrid(*axes, indexing="ij"), -1).reshape(-1, 4)

    def interp(self, V, s):
        """Multilinear V at s (N,4). Outside the box = failure: x/theta past the limits give l(s) < 0, and speeds past
        vmax / thdmax count as doomed (at 9 m/s the cart can't stop within the 4.8 m track; at 4 rad/s the pole can't be
        arrested within ±12°). Clamping speeds instead made V optimistic for fast carts (21% false-safe, all wall hits)."""
        u = (s - self.lo) / (self.hi - self.lo) * (self.n - 1)
        u = u.clamp(0, self.n - 1)
        i0 = u.floor().long().clamp(max=self.n - 2); w = u - i0
        out = torch.zeros(len(s), device=s.device)
        for corner in range(16):
            b = torch.tensor([(corner >> d) & 1 for d in range(4)], device=s.device)
            idx = i0 + b
            wt = torch.prod(torch.where(b.bool(), w, 1 - w), -1)
            flat = ((idx[:, 0] * self.n + idx[:, 1]) * self.n + idx[:, 2]) * self.n + idx[:, 3]
            out += wt * V[flat]
        fast = (s[:, 1].abs() > self.hi[1]) | (s[:, 3].abs() > self.hi[3])
        out = torch.where(fast, torch.clamp(margin(s), max=-1e-3), out)
        outside = (s[:, 0].abs() > X_MAX) | (s[:, 2].abs() > TH_MAX)
        return torch.where(outside, margin(s), out)


def rollouts(s, k):
    """All 2^k action sequences from s: (endpoints (2^k,N,4), path-min margin over steps 1..k (2^k,N), first action (2^k,))."""
    ends, mins, first = [], [], []
    for seq in range(2 ** k):
        x, m = s, None
        for j in range(k):
            x = step(x, torch.full((len(s),), (seq >> j) & 1, device=s.device))
            m = margin(x) if m is None else torch.minimum(m, margin(x))
        ends.append(x); mins.append(m); first.append(seq & 1)
    return torch.stack(ends), torch.stack(mins), torch.tensor(first, device=s.device)


def backup(g, V, s, k):
    """k-step Bellman backup: best sequence's min(path margin, V at endpoint); also its first action."""
    ends, mins, first = rollouts(s, k)
    q = torch.minimum(mins, g.interp(V, ends.reshape(-1, 4)).reshape(mins.shape))
    best, arg = q.max(0)
    return best, first[arg]


def solve(n=51, k=1, vmax=9.0, thdmax=4.0, iters=1000, tol=1e-6, dev="cuda", chunk=200_000):
    g = Grid(n, vmax, thdmax, dev)
    l = margin(g.pts)
    pre = [rollouts(g.pts[i:i + chunk], k) for i in range(0, len(g.pts), chunk)]  # fixed grid -> precompute
    V = l.clone()
    for it in range(iters):
        q = []
        for ends, mins, _ in pre:
            q.append(torch.minimum(mins, g.interp(V, ends.reshape(-1, 4)).reshape(mins.shape)).max(0).values)
        Vn = torch.minimum(l, torch.cat(q))
        d = (Vn - V).abs().max().item(); V = Vn
        if it % 50 == 0: print(f"  it={it} max|dV|={d:.2e} safe={float((V >= 0).float().mean()):.3f}", flush=True)
        if d < tol: break
    print(f"converged it={it} dV={d:.1e}")
    return g, V


def greedy(g, V, s, k=1):
    """First action of the best k-step sequence under V."""
    return backup(g, V, s, k)[1]


def check(g, V, k=1, n=20000, T=500, seed=0):
    """Sample states; does the greedy-V policy keep the safe-labelled ones alive for T steps, and does
    any of {greedy-V, always-left, always-right} save the doomed-labelled ones?"""
    gen = torch.Generator(device=V.device).manual_seed(seed)
    s0 = g.lo + (g.hi - g.lo) * torch.rand(n, 4, device=V.device, generator=gen)
    s0 = s0[margin(s0) >= 0]
    v0 = g.interp(V, s0)

    def survives(policy):
        s, alive = s0.clone(), torch.ones(len(s0), dtype=torch.bool, device=s0.device)
        for _ in range(T):
            s = step(s, policy(s)); alive &= margin(s) >= 0
        return alive
    sg = survives(lambda s: greedy(g, V, s, k))
    any_ = sg | survives(lambda s: torch.zeros(len(s), device=s.device)) | survives(lambda s: torch.ones(len(s), device=s.device))
    safe, doomed = v0 >= 0, v0 < 0
    print(f"states={len(s0)}  V>=0: {safe.float().mean():.3f}")
    print(f"  V>=0 & greedy survives {T}: {sg[safe].float().mean():.4f}   (|V|>.02: {sg[v0 > .02].float().mean():.4f})")
    print(f"  V<0  & any policy survives: {any_[doomed].float().mean():.4f}   (V<-.02: {any_[v0 < -.02].float().mean():.4f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=51)
    ap.add_argument("--k", type=int, default=1, help="lookahead steps per backup")
    ap.add_argument("--iters", type=int, default=1000)
    ap.add_argument("--out", default="/data/seongbin/lewm/datasets/cartpole_value.npz")
    a = ap.parse_args()
    g, V = solve(a.n, a.k, iters=a.iters)
    check(g, V, a.k)
    np.savez_compressed(a.out, V=V.reshape((a.n,) * 4).cpu().numpy(), lo=g.lo.cpu().numpy(), hi=g.hi.cpu().numpy(), k=a.k)
    print("saved", a.out)


def load(path, dev="cuda"):
    z = np.load(path)
    g = Grid(z["V"].shape[0], float(-z["lo"][1]), float(-z["lo"][3]), dev)
    return g, torch.as_tensor(z["V"].reshape(-1), device=dev), int(z["k"]) if "k" in z else 1
