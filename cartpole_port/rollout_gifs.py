"""Save example cartpole rollouts as GIFs (dataset policy: V-greedy + eps random actions). Banner shows the
state, margin l and value V; amber when doomed (l >= 0 > V), red once failed (l < 0)."""
import os, sys, argparse
import numpy as np, torch
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from value_dp import load, step, margin, greedy
from render import render

CASES = [  # (name, start state, eps)
    ("greedy", [-1.0, 0.5, 0.05, 0.0], 0.0),
    ("noisy_eps0.4", [0.5, 0.0, 0.0, 0.0], 0.4),
    ("doomed_start", [1.2, 2.5, 0.08, 0.6], 0.0),   # cart rushing at the right wall, pole leaning right
]


def frame(s, l, v, t):
    img = Image.fromarray(render(s)).resize((384, 384), Image.NEAREST)
    bar = Image.new("RGB", (384, 44), (40, 40, 40) if l >= 0 and v >= 0 else (200, 130, 0) if l >= 0 else (190, 30, 30))
    d = ImageDraw.Draw(bar)
    d.text((6, 4), f"t={t:3d}  x={s[0]:+.2f} xd={s[1]:+.2f} th={np.degrees(s[2]):+5.1f}deg thd={s[3]:+.2f}", fill="white")
    d.text((6, 24), f"margin l={l:+.3f}   value V={v:+.3f}   " + ("FAILED" if l < 0 else "DOOMED" if v < 0 else "safe"), fill="white")
    out = Image.new("RGB", (384, 428)); out.paste(bar, (0, 0)); out.paste(img, (0, 44))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--value", default="/data/seongbin/lewm/datasets/cartpole_value.npz")
    ap.add_argument("--out_dir", default="figs/cartpole_rollouts")
    ap.add_argument("--T", type=int, default=150)
    a = ap.parse_args()
    g, V, k = load(a.value)
    os.makedirs(a.out_dir, exist_ok=True)
    gen = torch.Generator(device=V.device).manual_seed(0)
    for name, s0, eps in CASES:
        s, frames, post = torch.tensor([s0], device=V.device), [], 0
        for t in range(a.T):
            l, v = margin(s).item(), g.interp(V, s).item()
            frames.append(frame(s[0].tolist(), l, v, t))
            if l < 0:
                post += 1
                if post > 10: break
            act = greedy(g, V, s, k)
            if torch.rand(1, device=V.device, generator=gen).item() < eps:
                act = torch.randint(0, 2, (1,), device=V.device, generator=gen)
            s = step(s, act)
        frames[0].save(f"{a.out_dir}/cartpole_{name}.gif", save_all=True, append_images=frames[1:], duration=40, loop=0)
        print(f"{name}: {len(frames)} frames, ended {'FAILED' if l < 0 else 'alive'}")


if __name__ == "__main__":
    main()
