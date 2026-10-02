import numpy as np
import torch
from stable_pretraining import data as dt
from lightning.pytorch.callbacks import Callback

def get_img_preprocessor(source: str, target: str, img_size: int = 224):
    imagenet_stats = dt.dataset_stats.ImageNet
    to_image = dt.transforms.ToImage(**imagenet_stats, source=source, target=target)
    resize = dt.transforms.Resize(img_size, source=source, target=target)
    return dt.transforms.Compose(to_image, resize)


class ZScoreNormalizer:
    """Picklable z-score normalizer — uses a class instead of a closure so it
    survives pickle when DataLoader workers are spawned (required by LanceDataset)."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, x):
        return ((x - self.mean) / self.std).float()


class ObstacleRecolor:
    """Recolor the (red) obstacle to a random TRAINING color per window, so the WM
    sees obstacles in many colors instead of only red — while purple/magenta are held
    OUT of the palette and stay OOD. Obstacle pixels are isolated by their "redness"
    a = relu(min(R-G, R-B))/255 (1.0 in the pure-red interior, fractional on the
    antialiased edges that blend red with the white background, 0 on green goal / blue
    agent / white bg). Remap toward the target color C by  out = x + a*(C - RED),
    which is exact for a red-on-white blend (RED and white share R=255), so edges
    recolor cleanly. Only obstacle color changes; geometry, goal, agent, background are
    untouched (color IS semantic here — cf the AugInvarianceReg caution in module.py).
    Input/return: (T, 3, H, W) uint8 tensor (one color per call = one per window).
    Picklable (survives DataLoader worker spawn), torch-RNG so workers differ.
    """
    RED = (255.0, 0.0, 0.0)
    # Non-purple palette. red is index 0 (kept in-distribution); the rest span
    # warm->earthy->teal, none in the purple/magenta region (high B, low G).
    DEFAULT_PALETTE = [
        (255, 0, 0), (255, 140, 0), (255, 200, 0), (150, 75, 0),
        (128, 128, 0), (250, 128, 114), (140, 0, 0), (0, 150, 150),
    ]
    # sg hazards are BLUE; recolor toward non-purple training colors (blue kept in-dist, index 0).
    BLUE = (0.0, 0.0, 255.0)
    SG_PALETTE = [
        (0, 0, 255), (255, 153, 0), (0, 153, 153), (255, 255, 0),
        (140, 69, 18), (0, 178, 0),
    ]

    def __init__(self, palette=None, p_apply=1.0, rotate=False, source="red"):
        # source: "red" (dubins obstacle) or "blue" (sg hazard) — which channel isolates the obstacle
        self.source = source
        if palette is None:
            palette = self.SG_PALETTE if source == "blue" else self.DEFAULT_PALETTE
        self.src_rgb = self.BLUE if source == "blue" else self.RED
        self.palette = palette
        self.p_apply = float(p_apply)
        # rotate: also apply a random k*90° whole-window rotation (matches the rot90
        # OOD transform). Valid for the image-only LEWM encoder with unchanged actions:
        # a globally-rotated rollout with the same angular velocity is a valid rollout,
        # and whole-scene rotation preserves the safety (relative agent-obstacle) label.
        self.rotate = bool(rotate)

    def __call__(self, px):
        out = px
        if float(torch.rand(())) <= self.p_apply:
            c = self.palette[int(torch.randint(len(self.palette), ()))]
            x = px.float()
            R, G, B = x[:, 0], x[:, 1], x[:, 2]
            if self.source == "blue":   # isolate blue-dominant (sg hazard) pixels
                a = torch.clamp(torch.minimum(B - R, B - G), min=0.0) / 255.0
            else:                        # isolate red-dominant (dubins obstacle) pixels
                a = torch.clamp(torch.minimum(R - G, R - B), min=0.0) / 255.0
            out = x.clone()
            for ci, (tv, rv) in enumerate(zip(c, self.src_rgb)):
                out[:, ci] = torch.clamp(x[:, ci] + a * (tv - rv), 0.0, 255.0)
            out = out.to(torch.uint8)
        if self.rotate:
            k = int(torch.randint(4, ()))            # one k per window, all T frames
            if k:
                out = torch.rot90(out, k, dims=(-2, -1)).contiguous()
        return out


def get_column_normalizer(dataset, source: str, target: str):
    """Get normalizer for a specific column in the dataset."""
    col_data = dataset.get_col_data(source)
    data = torch.from_numpy(np.array(col_data))
    data = data[~torch.isnan(data).any(dim=1)]
    mean = data.mean(0, keepdim=True).clone()
    std = data.std(0, keepdim=True).clone()
    return dt.transforms.WrapTorchTransform(ZScoreNormalizer(mean, std), source=source, target=target)

class SaveCkptCallback(Callback):
    """Callback to save model checkpoint after each epoch using save_pretrained."""

    def __init__(self, run_name, cfg, epoch_interval: int = 1):
        super().__init__()
        self.run_name = run_name
        self.cfg = cfg
        self.epoch_interval = epoch_interval

    def on_train_epoch_end(self, trainer, pl_module):
        super().on_train_epoch_end(trainer, pl_module)

        if trainer.is_global_zero:
            if (trainer.current_epoch + 1) % self.epoch_interval == 0:
                self._save(pl_module.model, trainer.current_epoch + 1)

            if (trainer.current_epoch + 1) == trainer.max_epochs:
                self._save(pl_module.model, trainer.current_epoch + 1)

    def _save(self, model, epoch):
        from stable_worldmodel.wm.utils import save_pretrained
        save_pretrained(
            model,
            run_name=self.run_name,
            config=self.cfg,
            filename=f'weights_epoch_{epoch}.pt',
        )