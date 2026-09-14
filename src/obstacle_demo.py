"""
Obstacle avoidance demo: NMPC weaving a straight "lane" around two static
obstacles, vs. the same NMPC with obstacle-avoidance turned off (which drives
straight through them) -- to make the effect of the avoidance constraint
obvious and measurable, not just visually plausible.

See NMPCController.solve() in nmpc_controller.py for the constraint itself:
a soft (squared-penalty) keep-out zone around each obstacle center, active
over the whole prediction horizon.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from nmpc_controller import NMPCConfig, NMPCController
from simulate import SimResult, _cross_track_error
from trajectory import cumulative_arclength, nearest_index, reference_horizon

OBSTACLES = [
    (40.0, 0.4, 1.5),    # (X, Y, radius) -- just left of lane center
    (70.0, -0.6, 1.3),   # just right of lane center
]
LANE_LENGTH = 100.0
V_TARGET = 8.0


def lane_path(n_points: int = 2000) -> np.ndarray:
    """A straight-line reference "lane": [X, Y=0, psi=0, v=V_TARGET]."""
    X = np.linspace(0, LANE_LENGTH, n_points)
    Y = np.zeros_like(X)
    psi = np.zeros_like(X)
    v = np.full_like(X, V_TARGET)
    return np.column_stack([X, Y, psi, v])


def run_with_obstacles(path: np.ndarray, model: DynamicBicycleModel, cfg: NMPCConfig,
                        sim_time: float, obstacles: list[tuple[float, float, float]] | None) -> SimResult:
    """Same pattern as simulate.run_nmpc, factored out here so the demo can
    toggle `obstacles` on/off for the same otherwise-identical run."""
    controller = NMPCController(model, cfg)
    n_steps = int(sim_time / model.dt)
    s = cumulative_arclength(path)

    state = np.array([path[0, 0], path[0, 1], path[0, 2], V_TARGET, 0.0, 0.0])
    states = [state.copy()]
    controls, lat_err, horizons, solve_times = [], [], [], []
    min_dist_to_obstacle = {tuple(o): np.inf for o in (obstacles or [])}

    prev_idx = None
    for _ in range(n_steps):
        idx = nearest_index(path, state[:2], prev_idx=prev_idx)
        prev_idx = idx
        ref_h = reference_horizon(path, s, idx, cfg.horizon, model.dt, v_ref=V_TARGET)

        u = controller.solve(state, ref_h, obstacles=obstacles)
        state = model.step(state, u)
        states.append(state.copy())
        controls.append(u.copy())
        lat_err.append(_cross_track_error(path, idx, state))
        horizons.append(controller.last_predicted_states)

        for o in (obstacles or []):
            d = np.hypot(state[0] - o[0], state[1] - o[1])
            key = tuple(o)
            min_dist_to_obstacle[key] = min(min_dist_to_obstacle[key], d)

        if idx >= len(path) - 5:
            break

    T = len(states)
    result = SimResult(
        t=np.arange(T) * model.dt,
        states=np.array(states),
        controls=np.array(controls),
        lateral_error=np.array(lat_err),
        heading_error=np.zeros(T - 1),
        predicted_horizons=horizons,
        controller_name="NMPC" + (" + obstacle avoidance" if obstacles else " (no avoidance)"),
        solve_times=np.array(solve_times) if solve_times else None,
    )
    return result, min_dist_to_obstacle


if __name__ == "__main__":
    import os

    model = DynamicBicycleModel(params=VehicleParams(), dt=0.1)
    cfg = NMPCConfig(horizon=15)
    path = lane_path()

    print("Running NMPC WITHOUT obstacle avoidance...")
    result_off, dists_off = run_with_obstacles(path, model, cfg, sim_time=15.0, obstacles=None)
    print("Running NMPC WITH obstacle avoidance...")
    result_on, dists_on = run_with_obstacles(path, model, cfg, sim_time=15.0, obstacles=OBSTACLES)

    print("\n--- Closest approach to each obstacle ---")
    for o in OBSTACLES:
        ox, oy, r = o
        d_off = dists_off.get(o, float("nan"))
        # dists_off keys come from tuple(o) with obstacles=None -> empty; recompute directly
        d_off = np.min(np.hypot(result_off.states[:, 0] - ox, result_off.states[:, 1] - oy))
        d_on = dists_on[tuple(o)]
        print(f"Obstacle at ({ox}, {oy}), radius {r} m:")
        print(f"  without avoidance: closest approach = {d_off:.2f} m "
              f"({'COLLISION' if d_off < r else 'clear'})")
        print(f"  with avoidance:    closest approach = {d_on:.2f} m "
              f"({'COLLISION' if d_on < r else 'clear'})")

    os.makedirs("../results", exist_ok=True)
    from visualize import animate_tracking, obstacle_avoidance_plot

    xlim = (-5, LANE_LENGTH + 5)
    ylim = (-3.2, 3.2)
    obstacle_avoidance_plot(path, result_on, result_off, OBSTACLES, "../results/obstacle_avoidance_plot.png")
    animate_tracking(path, result_on, "../results/obstacle_avoidance.gif",
                      title="NMPC weaving around static obstacles", stride=2,
                      obstacles=OBSTACLES, xlim=xlim, ylim=ylim,
                      equal_aspect=False, figsize=(11, 4.5))
    print("\nSaved obstacle_avoidance_plot.png and obstacle_avoidance.gif to ../results/")
