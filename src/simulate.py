"""
Closed-loop simulation harness.

Runs the kinematic bicycle model under either the MPC controller or the
pure-pursuit baseline (or both, for comparison) along a chosen reference
trajectory, logging state/control/error history.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from baseline_controller import PurePursuitConfig, PurePursuitController
from mpc_controller import MPCConfig, MPCController
from trajectory import cumulative_arclength, get_trajectory, nearest_index, reference_horizon
from vehicle_model import KinematicBicycleModel


@dataclass
class SimResult:
    t: np.ndarray
    states: np.ndarray          # (T, 4) -> X, Y, psi, v
    controls: np.ndarray        # (T-1, 2) -> a, delta
    lateral_error: np.ndarray   # (T,) signed-ish cross-track distance to nearest path point
    heading_error: np.ndarray   # (T,)
    predicted_horizons: list = field(default_factory=list)  # MPC only, for animation
    targets: list = field(default_factory=list)              # pure-pursuit only, for animation
    controller_name: str = ""
    solve_times: np.ndarray | None = None


def _cross_track_error(path: np.ndarray, idx: int, state: np.ndarray) -> float:
    X, Y, psi, _ = state
    px, py = path[idx, 0], path[idx, 1]
    # signed distance: positive = vehicle left of the path tangent
    path_psi = path[idx, 2]
    dx, dy = X - px, Y - py
    return -np.sin(path_psi) * dx + np.cos(path_psi) * dy


def run_mpc(path: np.ndarray, model: KinematicBicycleModel, mpc_cfg: MPCConfig,
            sim_time: float, x0: np.ndarray | None = None) -> SimResult:
    controller = MPCController(model, mpc_cfg)
    n_steps = int(sim_time / model.dt)
    s = cumulative_arclength(path)

    state = x0.copy() if x0 is not None else path[0].copy()
    states = [state.copy()]
    controls = []
    lat_err, head_err = [], []
    horizons = []
    solve_times = []

    import time as _time
    prev_idx = None
    for _ in range(n_steps):
        idx = nearest_index(path, state[:2], prev_idx=prev_idx)
        prev_idx = idx
        ref_h = reference_horizon(path, s, idx, mpc_cfg.horizon, model.dt, v_ref=path[idx, 3])

        t0 = _time.perf_counter()
        u = controller.solve(state, ref_h)
        solve_times.append(_time.perf_counter() - t0)

        state = model.step(state, u)
        states.append(state.copy())
        controls.append(u.copy())
        lat_err.append(_cross_track_error(path, idx, state))
        head_err.append(np.arctan2(np.sin(state[2] - path[idx, 2]), np.cos(state[2] - path[idx, 2])))
        horizons.append(controller.last_predicted_states)

        if idx >= len(path) - 5:
            break

    T = len(states)
    return SimResult(
        t=np.arange(T) * model.dt,
        states=np.array(states),
        controls=np.array(controls),
        lateral_error=np.array(lat_err),
        heading_error=np.array(head_err),
        predicted_horizons=horizons,
        controller_name="MPC",
        solve_times=np.array(solve_times),
    )


def run_pure_pursuit(path: np.ndarray, model: KinematicBicycleModel, pp_cfg: PurePursuitConfig,
                      sim_time: float, x0: np.ndarray | None = None) -> SimResult:
    controller = PurePursuitController(model, pp_cfg)
    n_steps = int(sim_time / model.dt)

    state = x0.copy() if x0 is not None else path[0].copy()
    states = [state.copy()]
    controls = []
    lat_err, head_err, targets = [], [], []
    solve_times = []

    import time as _time
    prev_idx = None
    for _ in range(n_steps):
        idx = nearest_index(path, state[:2], prev_idx=prev_idx)
        prev_idx = idx

        t0 = _time.perf_counter()
        u = controller.solve(state, path, idx)
        solve_times.append(_time.perf_counter() - t0)

        state = model.step(state, u)
        states.append(state.copy())
        controls.append(u.copy())
        lat_err.append(_cross_track_error(path, idx, state))
        head_err.append(np.arctan2(np.sin(state[2] - path[idx, 2]), np.cos(state[2] - path[idx, 2])))
        targets.append(controller.target_point)

        if idx >= len(path) - 5:
            break

    T = len(states)
    return SimResult(
        t=np.arange(T) * model.dt,
        states=np.array(states),
        controls=np.array(controls),
        lateral_error=np.array(lat_err),
        heading_error=np.array(head_err),
        targets=targets,
        controller_name="Pure Pursuit",
        solve_times=np.array(solve_times),
    )


if __name__ == "__main__":
    model = KinematicBicycleModel(wheelbase=2.7, dt=0.1)
    path = get_trajectory("figure_eight", v_target=5.0, scale=30.0)
    x0 = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0])

    mpc_result = run_mpc(path, model, MPCConfig(), sim_time=40.0, x0=x0.copy())
    pp_result = run_pure_pursuit(path, model, PurePursuitConfig(), sim_time=40.0, x0=x0.copy())

    print(f"MPC  : mean |lateral error| = {np.mean(np.abs(mpc_result.lateral_error)):.3f} m, "
          f"max = {np.max(np.abs(mpc_result.lateral_error)):.3f} m, "
          f"mean solve time = {np.mean(mpc_result.solve_times)*1000:.2f} ms")
    print(f"PP   : mean |lateral error| = {np.mean(np.abs(pp_result.lateral_error)):.3f} m, "
          f"max = {np.max(np.abs(pp_result.lateral_error)):.3f} m, "
          f"mean solve time = {np.mean(pp_result.solve_times)*1000:.2f} ms")
