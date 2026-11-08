"""Sim-to-real style shifts for DMC Reacher, shared by reacher_ood.py (SHIFTS=...) and reacher_cem.py (SHIFT=...).
Severities are fixed a priori (moderate, not tuned on results).
  physics (render):  cam_shift (fixed camera +3 cm in x and y), cam_tilt (5 deg about x), cam_zoom (camera 10 cm closer),
                     light_dir (overhead light moved to (0.4, 0.4, 1), aimed at the origin)
  image:             noise (Gaussian, sigma 10/255), blur (Gaussian, sigma 1.5 px), jpeg (quality 15), exposure (gamma 1.5),
                     white_balance (R x1.2, B x0.8), occlusion (fixed 48 px grey square per env, seeded)
  dynamics (CEM):    gear0.7 (actuator gear x0.7), damping10 (joint damping x10), delay1 (1-env-step action latency),
                     act_noise (Gaussian action noise, sigma 0.2, clipped)"""
import io
import numpy as np
from PIL import Image, ImageFilter

PHYS = {"cam_shift", "cam_tilt", "cam_zoom", "light_dir", "gear0.7", "damping10"}
IMAGE = {"noise", "blur", "jpeg", "exposure", "white_balance", "occlusion"}
DYN = {"gear0.7", "damping10", "delay1", "act_noise"}
APPEARANCE = ["cam_shift", "cam_tilt", "cam_zoom", "light_dir", "noise", "blur", "jpeg", "exposure", "white_balance", "occlusion"]


def apply_physics(env, name):
    """Edit the MuJoCo model in place (after env.reset)."""
    m = env.env.physics.model
    if name == "cam_shift": m.cam_pos[0, :2] += 0.03
    elif name == "cam_tilt":
        t = np.deg2rad(5) / 2; m.cam_quat[0] = [np.cos(t), np.sin(t), 0, 0]
    elif name == "cam_zoom": m.cam_pos[0, 2] -= 0.10
    elif name == "light_dir":
        p = np.array([0.4, 0.4, 1.0]); m.light_pos[0] = p; m.light_dir[0] = -p / np.linalg.norm(p)
    elif name == "gear0.7": m.actuator_gear[:, 0] *= 0.7
    elif name == "damping10": m.dof_damping[:] *= 10


def corrupt(img, name, j=0):
    """img: (H, W, 3) uint8 -> uint8. j seeds per-env randomness (occlusion position, noise)."""
    if name not in IMAGE: return img
    r = np.random.default_rng(1000 + j)
    if name == "noise": return np.clip(img + r.normal(0, 10, img.shape), 0, 255).astype(np.uint8)
    if name == "blur": return np.asarray(Image.fromarray(img).filter(ImageFilter.GaussianBlur(1.5)))
    if name == "jpeg":
        b = io.BytesIO(); Image.fromarray(img).save(b, format="JPEG", quality=15); return np.asarray(Image.open(b).convert("RGB"))
    if name == "exposure": return (255 * (img / 255.) ** 1.5).astype(np.uint8)
    if name == "white_balance": return np.clip(img * np.array([1.2, 1.0, 0.8]), 0, 255).astype(np.uint8)
    if name == "occlusion":
        out = img.copy(); y, x = r.integers(0, img.shape[0] - 48, 2); out[y:y + 48, x:x + 48] = 128; return out


if __name__ == "__main__":   # self-check: every image corruption keeps shape/dtype and changes the frame
    x = np.random.default_rng(0).integers(0, 256, (224, 224, 3)).astype(np.uint8)
    for n in IMAGE:
        y = corrupt(x, n); assert y.shape == x.shape and y.dtype == np.uint8 and (y != x).any(), n
    print("ok")
