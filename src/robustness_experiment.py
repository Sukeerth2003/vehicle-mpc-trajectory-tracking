"""
Robustness experiment: how much does each controller's tracking quality
degrade when the *true* plant doesn't quite match what the controller thinks
it's controlling?

Three disturbances are injected into the true-plant simulation, invisible to
every controller (they only ever see/act on the disturbed measurement or the
disturbed state -- never the "clean" ground truth):

  1. Crosswind: a constant lateral drift added to the vehicle's world-frame
     position every step, representing a steady sideways push the controller
     has to continuously counter-steer against. (Simplification: modeled as a
     direct position drift rather than a force integrated through the vehicle
     dynamics -- documented here rather than overclaiming physical fidelity.)
  2. Process noise: small Gaussian noise on the true state's heading and
     speed every step, representing unmodeled dynamics / actuation noise.
  3. Sensor noise: Gaussian noise added to the state each controller
     *measures* (and plans against) -- the true plant integrates forward from
     the real state, but the controller only ever sees a noisy estimate of it.

All three controllers built so far -- LTV-MPC (kinematic model), NMPC
(dynamic model), and pure pursuit (run against both models) -- are compared
under the identical disturbance realization (same RNG seed per trial, so
differences reflect the controller, not luck), across several trials with
different seeds (a small Monte Carlo study) to separate real
robustness differences from noise-realization luck.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from baseline_controller import PurePursuitConfig, PurePursuitController
from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from mpc_controller import MPCConfig, MPCController
from nmpc_controller import NMPCConfig, NMPCController
from simulate import _cross_track_error
from trajectory import cumulative_arclength, get_trajectory, nearest_index, reference_horizon
from vehicle_model import KinematicBicycleModel


@dataclass
class DisturbanceConfig:
    crosswind_speed: float = 0.25     # m/s, constant world-frame +Y drift added to position
    process_noise_heading_std: float = 0.004   # rad, per step
    process_noise_speed_std: float = 0.03      # m/s, per step
    sensor_noise_pos_std: float = 0.05         # m
    sensor_noise_heading_std: float = 0.008    # rad
    sensor_noise_speed_std: float = 0.08       # m/s


def _sense(true_state: np.ndarray, rng: np.random.Generator, cfg: DisturbanceConfig) -> np.ndarray:
    """Noisy measurement of true_state, as seen by a controller."""
    noise = np.zeros_like(true_state)
    noise[0] = rng.normal(0, cfg.sensor_noise_pos_std)
    noise[1] = rng.normal(0, cfg.sensor_noise_pos_std)
    noise[2] = rng.normal(0, cfg.sensor_noise_heading_std)
    noise[3] = rng.normal(0, cfg.sensor_noise_speed_std)
    return true_state + noise


def _apply_plant_disturbance(true_state: np.ndarray, dt: float, rng: np.random.Generator,
                              cfg: DisturbanceConfig) -> np.ndarray:
    """Crosswind (position drift) + process noise, applied to the true plant
    state after each physics step -- never seen directly by any controller."""
    s = true_state.copy()
    s[1] += cfg.crosswind_speed * dt                       # crosswind drift (world +Y)
    s[2] += rng.normal(0, cfg.process_noise_heading_std)    # heading process noise
    s[3] += rng.normal(0, cfg.process_noise_speed_std)      # speed process noise
    return s


def _run_trial(controller_name: str, path: np.ndarray, sim_time: float, dt: float,
                dist_cfg: DisturbanceConfig, seed: int) -> float:
    """Returns mean |cross-track error| for one trial. controller_name in
    {'ltv_mpc', 'nmpc', 'pure_pursuit_kinematic', 'pure_pursuit_dynamic'}."""
    rng = np.random.default_rng(seed)
    s = cumulative_arclength(path)
    n_steps = int(sim_time / dt)

    if controller_name in ("ltv_mpc", "pure_pursuit_kinematic"):
        model = KinematicBicycleModel(wheelbase=2.7, dt=dt)
        true_state = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0])
    else:
        model = DynamicBicycleModel(params=VehicleParams(), dt=dt)
        true_state = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0, 0.0, 0.0])

    if controller_name == "ltv_mpc":
        cfg = MPCConfig()
        controller = MPCController(model, cfg)
        horizon = cfg.horizon
    elif controller_name == "nmpc":
        cfg = NMPCConfig(horizon=10)
        controller = NMPCController(model, cfg)
        horizon = cfg.horizon
    else:
        controller = PurePursuitController(model, PurePursuitConfig())
        horizon = None

    errs = []
    prev_idx = None
    for _ in range(n_steps):
        idx = nearest_index(path, true_state[:2], prev_idx=prev_idx)
        prev_idx = idx
        measured_state = _sense(true_state, rng, dist_cfg)

        if horizon is not None:
            ref_h = reference_horizon(path, s, idx, horizon, dt, v_ref=path[idx, 3])
            u = controller.solve(measured_state, ref_h)
        else:
            u = controller.solve(measured_state, path, idx)

        true_state = model.step(true_state, u)
        true_state = _apply_plant_disturbance(true_state, dt, rng, dist_cfg)

        errs.append(_cross_track_error(path, idx, true_state))

        if idx >= len(path) - 5:
            break

    return float(np.mean(np.abs(errs)))


def run_robustness_study(n_trials: int = 6, sim_time: float = 20.0, dt: float = 0.1):
    path = get_trajectory("double_lane_change", v_target=8.0)
    dist_cfg = DisturbanceConfig()
    controllers = ["ltv_mpc", "nmpc", "pure_pursuit_kinematic", "pure_pursuit_dynamic"]

    # Fixed (not Python's randomized hash()) per-controller seed offset, so the
    # same disturbance realization is used for a given controller across runs
    # -- reproducible, and each controller still gets an independent noise
    # sequence rather than literally the same one every trial.
    seed_offset = {"ltv_mpc": 11, "nmpc": 23, "pure_pursuit_kinematic": 37, "pure_pursuit_dynamic": 41}

    results = {name: [] for name in controllers}
    for trial in range(n_trials):
        for name in controllers:
            err = _run_trial(name, path, sim_time, dt, dist_cfg, seed=1000 * trial + seed_offset[name])
            results[name].append(err)
            print(f"trial {trial}  {name:24s}  mean|err| = {err:.4f} m")

    print("\n--- Summary (mean +/- std over trials, mean |cross-track error| [m]) ---")
    summary = {}
    for name in controllers:
        vals = np.array(results[name])
        summary[name] = (float(vals.mean()), float(vals.std()))
        print(f"{name:24s}  {vals.mean():.4f} +/- {vals.std():.4f} m")

    return results, summary


if __name__ == "__main__":
    import json
    import os

    results, summary = run_robustness_study()

    os.makedirs("../results", exist_ok=True)
    with open("../results/robustness_summary.json", "w") as f:
        json.dump({"results": results, "summary": summary}, f, indent=2)
    print("\nSaved ../results/robustness_summary.json")
