"""State -> 128x128 frame for gym CartPole, drawn directly (PIL, 4x supersampled) so colors are a clean
parameter: gym's own render is anti-aliased, so the swm exact-match color swap leaves the original color on
edge pixels. Cart matches gym (0.4x0.24 m), walls drawn at x=±2.4. ponytail: the pole is DRAWN 2.5 m long
(physics keeps 1.0 m) and the view is x in [-3,3], y in [-0.5,3] (angle exaggerated 1.7x), else a ±12°
lean is ~4 px at 128 px wide; a fixed bijective render choice, not a physics change."""
import numpy as np
from PIL import Image, ImageDraw

S, OUT = 512, 128
XV, YV = (-3.0, 3.0), (-0.5, 3.0)
POLE_VIS_LEN, POLE_VIS_HW = 2.5, 0.07
# in-distribution = gym's colors. Held-out OOD pole color: purple.
DEFAULT = dict(bg=(255, 255, 255), cart=(0, 0, 0), pole=(202, 152, 101), axle=(129, 132, 203),
               track=(0, 0, 0), wall=(150, 150, 150))
POLE_PALETTE = dict(tan=(202, 152, 101), red=(220, 40, 40), green=(40, 170, 60), blue=(40, 90, 220),
                    orange=(240, 140, 20), teal=(20, 160, 160), purple=(140, 50, 190))


def _px(x, y):
    return ((x - XV[0]) / (XV[1] - XV[0]) * S, (1 - (y - YV[0]) / (YV[1] - YV[0])) * S)


def render(state, colors=None):
    c = {**DEFAULT, **(colors or {})}
    x, _, th, _ = (float(v) for v in state)
    im = Image.new("RGB", (S, S), c["bg"]); d = ImageDraw.Draw(im)
    d.line([_px(XV[0], 0), _px(XV[1], 0)], fill=c["track"], width=4)
    for w in (-2.4, 2.4):
        d.line([_px(w, -0.25), _px(w, 0.6)], fill=c["wall"], width=8)
    cw, ch = 0.4, 0.24
    d.polygon([_px(x - cw / 2, 0), _px(x + cw / 2, 0), _px(x + cw / 2, ch), _px(x - cw / 2, ch)], fill=c["cart"])
    ax, ay = x, ch * 0.75  # axle, as in gym (cartheight/4 below cart top)
    u, n = np.array([np.sin(th), np.cos(th)]), np.array([np.cos(th), -np.sin(th)])  # along / across pole
    hw = POLE_VIS_HW
    corners = [np.array([ax, ay]) + a * u + b * hw * n for a, b in ((0, -1), (0, 1), (POLE_VIS_LEN, 1), (POLE_VIS_LEN, -1))]
    d.polygon([_px(*p) for p in corners], fill=c["pole"])
    r = 0.06; (px, py), (qx, qy) = _px(ax - r, ay + r), _px(ax + r, ay - r)
    d.ellipse([px, py, qx, qy], fill=c["axle"])
    return np.asarray(im.resize((OUT, OUT), Image.BOX))


if __name__ == "__main__":
    import sys
    rows = [np.concatenate([render(s, dict(pole=POLE_PALETTE[k])) for s in
                            ([0, 0, 0, 0], [2.3, 0, .2, 0], [-1.5, 0, -.15, 0], [0.5, 0, .6, 0])], 1)
            for k in ("tan", "purple")]
    Image.fromarray(np.concatenate(rows, 0)).save(sys.argv[1] if len(sys.argv) > 1 else "render_check.png")
