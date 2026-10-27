"""Grid of the cartpole OOD conditions: rows = condition, columns = the SAME test states (2 safe, 2 doomed,
2 failed), so each column differs only by the perturbation. Labels unchanged across rows."""
import os, sys
import numpy as np, h5py
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render import render, POLE_PALETTE, VIEWS

OUT = sys.argv[1] if len(sys.argv) > 1 else "figs/cartpole_rollouts/cartpole_ood_samples.png"
CONDS = [("in-dist (tan)", {}), ("purple pole", dict(colors=dict(pole=POLE_PALETTE["purple"]))),
         *((k, dict(view=v)) for k, v in VIEWS.items())]
with h5py.File("/data/seongbin/lewm/datasets/cartpole_expert.h5") as f:
    s, m, v, ep = f["state"][:], f["margin"][:], f["value"][:], f["episode_idx"][:]
rng = np.random.default_rng(1)
te = ep % 10 >= 7
pick = [(lbl, i) for lbl, sel in (("safe", v >= 0), ("doomed", (m >= 0) & (v < 0)), ("failed", m < 0))
        for i in rng.choice(np.where(te & sel)[0], 2, replace=False)]
T, L, P = 256, 120, 4                                     # tile px, left label width, padding
img = Image.new("RGB", (L + len(pick) * (T + P), 30 + len(CONDS) * (T + P)), "white"); d = ImageDraw.Draw(img)
for j, (lbl, i) in enumerate(pick):
    d.text((L + j * (T + P) + 4, 4), f"{lbl}  x={s[i, 0]:+.2f} xd={s[i, 1]:+.1f}", fill="black")
    d.text((L + j * (T + P) + 4, 16), f"th={np.degrees(s[i, 2]):+.1f}deg  V={v[i]:+.2f}", fill="black")
for r, (name, kw) in enumerate(CONDS):
    y = 30 + r * (T + P)
    d.text((4, y + T // 2), name, fill="black")
    for j, (_, i) in enumerate(pick):
        img.paste(Image.fromarray(render(s[i], **kw)).resize((T, T), Image.NEAREST), (L + j * (T + P), y))
os.makedirs(os.path.dirname(OUT), exist_ok=True); img.save(OUT); print("saved", OUT)
