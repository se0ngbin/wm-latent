"""
Controller configuration for DubinsEnv.

Only MPPI and diffusion policies are supported (including ``diffusion_wm`` for
world-model–filtered diffusion).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

from .paths import DIFFUSION_CHECKPOINT, DIFFUSION_DIR

ControllerName = Literal["mppi", "diffusion", "diffusion_wm", "lewm_planner"]


@dataclass
class ControllerConfig:
    """Parameters for MPPI or diffusion controllers."""

    controller_type: ControllerName = "mppi"

    # --- MPPI (used when controller_type == "mppi") ---
    prediction_horizon: int = 10
    num_samples: int = 50
    temperature: float = 2.0
    lambda_param: float = 0.5
    noise_variance: float = 2.0
    adaptive_temperature: bool = True
    warm_start: bool = True
    goal_weight: float = 10.0
    obstacle_weight: float = 50.0
    control_weight: float = 0.1
    obstacle_safety_margin: float = 0.3
    goal_tolerance: float = 0.1

    # --- Diffusion / diffusion_wm (same checkpoint fields; WM is toggled via DreamerConfig) ---
    checkpoint_path: str = str(DIFFUSION_CHECKPOINT)
    config_path: str = f"{DIFFUSION_DIR}/"
    checkpoint_version: int = 1000
    device: str = "cuda:0"
    action_chunk_size: int = 8
    total_chunk_size: int = 16
    eval_diffusion_steps: int = 16

    # --- LE-WM planner (used when controller_type == "lewm_planner") ---
    # Checkpoint relative to $STABLEWM_HOME/checkpoints (e.g. "sigreg_only_dubins/weights_epoch_50.pt").
    lewm_ckpt: str = "sigreg_only_dubins/weights_epoch_50.pt"
    lewm_cache_dir: Optional[str] = None  # None => $STABLEWM_HOME
    lewm_horizon: int = 12
    lewm_num_samples: int = 200
    lewm_n_iters: int = 4
    lewm_topk: int = 20
    lewm_var_scale: float = 1.0
    lewm_replan_every: int = 1
    # Safe-by-construction action map (see controllers/safe_action_map.py).
    lewm_safe_mode: str = "off"  # "off" | "learned" | "gt"
    lewm_safe_backend: str = "grid"  # "grid" | "hardnet"
    lewm_hardnet_iters: int = 3
    lewm_hardnet_damping: float = 1.0
    lewm_margin_ckpt: Optional[str] = None
    lewm_margin_head: str = "margin_gp"
    lewm_cbf_alpha: float = 0.3
    lewm_safe_grid: int = 21
    lewm_gt_h: str = "hj"  # "hj" | "dist" (gt mode only)
    lewm_h_offset: float = 0.0
    lewm_penalty_weight: float = 0.0

    seed: Optional[int] = None
