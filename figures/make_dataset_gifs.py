"""GIFs of stored training frames (no re-rendering): red border = failure frame, green = safe.
Dubins: one MPPI demonstration and one colliding diffusion rollout (whole episodes).
Point: 8-frame chunks only (episodes are stored chunked and shuffled), several chunks played back to back.
Usage: python make_dataset_gifs.py   (writes next to this file)"""
import os, h5py, numpy as np
from PIL import Image, ImageDraw

OUT = os.path.dirname(os.path.abspath(__file__))


def frame(img, fail, text, scale):
    im = Image.fromarray(img).resize((img.shape[1] * scale, img.shape[0] * scale), Image.NEAREST)
    b = 6
    canvas = Image.new("RGB", (im.width + 2 * b, im.height + 2 * b + 18), (255, 255, 255))
    ImageDraw.Draw(canvas).rectangle([0, 0, im.width + 2 * b - 1, im.height + 2 * b - 1],
                                     fill=(220, 30, 30) if fail else (40, 170, 70))
    canvas.paste(im, (b, b))
    ImageDraw.Draw(canvas).text((b, im.height + 2 * b + 3), text, fill=(0, 0, 0))
    return canvas


def save(frames, name, ms):
    frames[0].save(f"{OUT}/{name}", save_all=True, append_images=frames[1:], duration=ms, loop=0)
    print("wrote", name, len(frames), "frames")


# ---- Dubins ----
f = h5py.File("/data/seongbin/lewm/datasets/dubins_expert.h5", "r")
off, ln, fl = f["ep_offset"][:], f["ep_len"][:], f["failures"][:].astype(bool)
coll = lambda e: fl[off[e]:off[e] + ln[e]].any()
mppi = next(e for e in range(3900, 4100) if not coll(e))
diff = next(e for e in range(2000, 3900) if 0.2 < fl[off[e]:off[e] + ln[e]].mean() < 0.45)
for e, tag in ((mppi, "mppi"), (diff, "diffusion")):
    px = f["pixels"][off[e]:off[e] + ln[e]]
    fr = [frame(px[i], fl[off[e] + i], f"Dubins {tag} ep {e}  step {i}/{ln[e]}", 2) for i in range(ln[e])]
    save(fr, f"dubins_{tag}_ep{e}.gif", 80)

# ---- Point ----
g = h5py.File("/data/seongbin/lewm/datasets/sg_point_ovh224.h5", "r")
pf = g["failures"][:].astype(bool).reshape(-1, 8)
sp = np.linalg.norm(g["velocity"][:], axis=1).reshape(-1, 8)
rng = np.random.default_rng(0)
pick = lambda m: int(rng.choice(np.where(m)[0]))
chunks = [("safe, fast", pick(~pf.any(1) & (sp.mean(1) > 1.0))),
          ("enters hazard", pick(~pf[:, 0] & pf[:, -1])),
          ("inside, slow", pick(pf.all(1) & (sp.mean(1) < 0.15))),
          ("crosses at speed", pick(pf[:, 2:6].any(1) & ~pf[:, 0] & ~pf[:, -1] & (sp.mean(1) > 0.8))),
          ("leaves hazard", pick(pf[:, 0] & ~pf[:, -1]))]
fr = []
for tag, c in chunks:
    px = g["pixels"][c * 8:(c + 1) * 8]
    fr += [frame(px[i], pf[c, i], f"{tag}  t={i}/8  |v|={sp[c, i]:.2f}", 1) for i in range(8)]
save(fr, "point_chunks.gif", 250)
