"""Solve the ∞-horizon avoid HJ-PDE for the 3D Dubins env on a grid; save V_truth(x,y,θ).

Dynamics (matches src/latent_cbf/dubins/dubins_env.py with default config):
    ẋ = cos(θ),  ẏ = sin(θ),  θ̇ = u,   u ∈ [-2.0, 2.0]

Constraint (failure set = obstacles): two circles centered at (0.25, ±0.65), r=0.5.
ℓ(x,y) = min over obstacles of (distance(agent, center) - radius)  → positive outside obstacles.

The avoid value function V*(x) = sup_u inf_{t≥0} ℓ(x(t)) is the largest signed safety
margin reachable from x; V* ≥ 0 means a safe trajectory exists (state x is in the safe set).

We solve via the time-discounted HJ PDE
    ∂V/∂t = -min{ V - ℓ,   max_u ∇V·f(x,u) }
running to convergence (or fixed horizon). Uses hj_reachability (JAX-based) with a
custom 3-D dynamics class.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

import hj_reachability as hj
from hj_reachability import dynamics as hj_dynamics
from hj_reachability import sets as hj_sets


class Dubins3D(hj_dynamics.ControlAndDisturbanceAffineDynamics):
    """Dubins car with fixed forward speed, control = angular velocity (no disturbance)."""

    def __init__(self, speed: float = 1.0, max_turn_rate: float = 2.0,
                 control_mode: str = "max"):
        self.speed = speed
        control_space = hj_sets.Box(jnp.array([-max_turn_rate]), jnp.array([max_turn_rate]))
        disturbance_space = hj_sets.Box(jnp.array([0.0]), jnp.array([0.0]))
        super().__init__(control_mode=control_mode, disturbance_mode="min",
                         control_space=control_space, disturbance_space=disturbance_space)

    def open_loop_dynamics(self, state, time):
        x, y, theta = state
        return jnp.array([self.speed * jnp.cos(theta),
                          self.speed * jnp.sin(theta),
                          0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[0.0], [0.0], [1.0]])

    def disturbance_jacobian(self, state, time):
        return jnp.zeros((3, 1))


def signed_distance_obstacles(grid_xyz, obstacles):
    """For each grid point, ℓ(x,y) = min over obstacles of (||(x,y)-(cx,cy)|| - r).
    grid_xyz: (nx, ny, ntheta, 3); we ignore theta."""
    x = grid_xyz[..., 0]
    y = grid_xyz[..., 1]
    sdfs = []
    for (cx, cy, r) in obstacles:
        sdfs.append(jnp.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r)
    return jnp.min(jnp.stack(sdfs, axis=-1), axis=-1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=81)
    parser.add_argument("--ny", type=int, default=81)
    parser.add_argument("--ntheta", type=int, default=64)
    parser.add_argument("--x_min", type=float, default=-1.5)
    parser.add_argument("--x_max", type=float, default=1.5)
    parser.add_argument("--y_min", type=float, default=-1.5)
    parser.add_argument("--y_max", type=float, default=1.5)
    parser.add_argument("--max_turn_rate", type=float, default=2.0)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--horizon", type=float, default=3.0,
                        help="Solve over t ∈ [0, horizon] backwards; longer = closer to ∞-horizon.")
    parser.add_argument("--n_steps", type=int, default=120,
                        help="Number of substeps in time integration.")
    parser.add_argument("--obstacles", nargs="*", default=["0.25,0.65,0.5", "0.25,-0.65,0.5"])
    parser.add_argument("--out", default="/home/seongbin/latent/latent_cbf/results/hj_truth.npz")
    args = parser.parse_args()

    obstacles = [tuple(float(v) for v in s.split(",")) for s in args.obstacles]

    # Grid
    domain = hj_sets.Box(jnp.array([args.x_min, args.y_min, -jnp.pi]),
                        jnp.array([args.x_max, args.y_max, jnp.pi]))
    grid = hj.Grid.from_lattice_parameters_and_boundary_conditions(
        domain, (args.nx, args.ny, args.ntheta),
        periodic_dims=2,
    )

    # Initial value: signed-distance to obstacles (positive outside).
    obstacles_jnp = [tuple(float(v) for v in o) for o in obstacles]
    values_0 = signed_distance_obstacles(grid.states, obstacles_jnp)

    # Dynamics: evader (control_mode='max' → V increases under best safe action).
    dyn = Dubins3D(speed=args.speed, max_turn_rate=args.max_turn_rate, control_mode="max")

    # Solver: standard upwind with global Lax-Friedrichs Hamiltonian; avoid postprocessor.
    solver_settings = hj.SolverSettings.with_accuracy(
        "very_high",
        hamiltonian_postprocessor=hj.solver.backwards_reachable_tube,
    )

    print(f"Grid: {args.nx}×{args.ny}×{args.ntheta} = {args.nx*args.ny*args.ntheta:,} cells")
    print(f"Time horizon: 0 → -{args.horizon}, n_steps = {args.n_steps}")
    print(f"Obstacles: {obstacles}")

    t0 = time.time()
    times = jnp.linspace(0.0, -args.horizon, args.n_steps + 1)
    values = hj.solve(solver_settings, dyn, grid, times, values_0)
    values = np.asarray(values[-1])  # final = converged value at t = -horizon
    elapsed = time.time() - t0
    print(f"Solve done in {elapsed:.1f}s. V range: [{values.min():.3f}, {values.max():.3f}]")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        out,
        V=values.astype(np.float32),
        grid_xs=np.linspace(args.x_min, args.x_max, args.nx, dtype=np.float32),
        grid_ys=np.linspace(args.y_min, args.y_max, args.ny, dtype=np.float32),
        grid_thetas=np.linspace(-np.pi, np.pi, args.ntheta, endpoint=False, dtype=np.float32),
        obstacles=np.asarray(obstacles, dtype=np.float32),
        max_turn_rate=np.float32(args.max_turn_rate),
        speed=np.float32(args.speed),
        horizon=np.float32(args.horizon),
    )
    safe_frac = float((values >= 0).mean())
    print(f"saved {out}  (safe-set fraction = {safe_frac:.3f})")


if __name__ == "__main__":
    main()
