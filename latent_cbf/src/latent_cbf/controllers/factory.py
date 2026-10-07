"""
Controller factory for creating controllers from configuration.
"""

from __future__ import annotations

from typing import Optional, Union

from configs import Config
from configs.dreamer_conf import DreamerConfig

from .mppi_controller import MPPIController
from .diffusion_controller import DiffusionController, FilteredDiffusionController


def _make_diffusion_controller(ctrl_config) -> DiffusionController:
    return DiffusionController(
        checkpoint_path=ctrl_config.checkpoint_path,
        config_path=ctrl_config.config_path,
        device=ctrl_config.device,
        action_chunk_size=ctrl_config.action_chunk_size,
        total_chunk_size=ctrl_config.total_chunk_size,
        eval_diffusion_steps=ctrl_config.eval_diffusion_steps,
    )


def create_controller_from_config(
    config: Config,
    wm_config: Optional[DreamerConfig] = None,
) -> Union[MPPIController, DiffusionController, FilteredDiffusionController]:
    """
    Create a controller from configuration.

    For ``diffusion`` / ``diffusion_wm``, builds a :class:`DiffusionController` and
    wraps it with :class:`FilteredDiffusionController` when ``wm_config`` is given
    and ``wm_config.use_wm_prediction`` is True.
    """
    ctrl_config = config.controller
    ctype = ctrl_config.controller_type

    if ctype == "mppi":
        return MPPIController(
            prediction_horizon=ctrl_config.prediction_horizon,
            num_samples=ctrl_config.num_samples,
            dt=config.environment.dt,
            speed=config.environment.speed,
            max_angular_velocity=config.environment.max_angular_velocity,
            temperature=ctrl_config.temperature,
            lambda_param=ctrl_config.lambda_param,
            goal_weight=ctrl_config.goal_weight,
            obstacle_weight=ctrl_config.obstacle_weight,
            control_weight=ctrl_config.control_weight,
            obstacle_safety_margin=ctrl_config.obstacle_safety_margin,
            goal_tolerance=ctrl_config.goal_tolerance,
            noise_variance=ctrl_config.noise_variance,
            adaptive_temperature=ctrl_config.adaptive_temperature,
            warm_start=ctrl_config.warm_start,
        )
    if ctype in ("diffusion", "diffusion_wm"):
        base = _make_diffusion_controller(ctrl_config)
        if wm_config is not None and getattr(wm_config, "use_wm_prediction", False):
            return FilteredDiffusionController(base, wm_config)
        return base

    if ctype == "lewm_planner":
        from .lewm_planner_controller import LeWMPlannerController

        return LeWMPlannerController(
            lewm_ckpt=ctrl_config.lewm_ckpt,
            cache_dir=ctrl_config.lewm_cache_dir,
            device=ctrl_config.device,
            horizon=ctrl_config.lewm_horizon,
            num_samples=ctrl_config.lewm_num_samples,
            n_iters=ctrl_config.lewm_n_iters,
            topk=ctrl_config.lewm_topk,
            var_scale=ctrl_config.lewm_var_scale,
            seed=ctrl_config.seed or 0,
            safe_mode=getattr(ctrl_config, "lewm_safe_mode", "off"),
            safe_backend=getattr(ctrl_config, "lewm_safe_backend", "grid"),
            hardnet_iters=getattr(ctrl_config, "lewm_hardnet_iters", 3),
            hardnet_damping=getattr(ctrl_config, "lewm_hardnet_damping", 1.0),
            margin_ckpt=getattr(ctrl_config, "lewm_margin_ckpt", None),
            margin_head=getattr(ctrl_config, "lewm_margin_head", "margin_gp"),
            cbf_alpha=getattr(ctrl_config, "lewm_cbf_alpha", 0.3),
            safe_grid=getattr(ctrl_config, "lewm_safe_grid", 21),
            gt_h=getattr(ctrl_config, "lewm_gt_h", "hj"),
            h_offset=getattr(ctrl_config, "lewm_h_offset", 0.0),
            penalty_weight=getattr(ctrl_config, "lewm_penalty_weight", 0.0),
        )

    raise ValueError(
        f"Unknown controller_type {ctype!r}; expected 'mppi', 'diffusion', "
        "'diffusion_wm', or 'lewm_planner'."
    )
