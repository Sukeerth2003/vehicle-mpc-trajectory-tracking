"""
Part 5: the comprehensive ablation-matrix comparison this extension is built
around -- {kinematic model + LTV-MPC, dynamic model + NMPC} x {naive, CV,
CTRV, unimodal SSM, multimodal SSM (scenario-based)}, run on two
standardized moving-pedestrian scenarios:

  - "moving_stop"    -- Part 3's deterministic stopping pedestrian
                         (moving_obstacle_demo.py): unambiguous, just fast.
  - "ambiguous_go"/"ambiguous_stop" -- Part 4's genuinely ambiguous
                         go-or-stop pedestrian (multimodal_obstacle_demo.py),
                         with the true outcome FORCED so both branches get
                         equal, matched trials.

Every controller/method pair in a given (scenario, trial) sees the EXACT
SAME true pedestrian trajectory and sensor-noise realization -- the same
matched-comparison discipline as robustness_experiment.py and the Part 3/4
demos -- so differences in the results are attributable to the
controller/predictor combination, not to which trial happened to be easier.

This deliberately reuses each scenario's already-built-and-validated ground
truth generator and lane geometry from moving_obstacle_demo.py and
multimodal_obstacle_demo.py rather than reimplementing them; what's new here
is comparing BOTH controllers against ALL FIVE prediction methods on both
scenario families in one place, plus the kinematic+LTV-MPC controller
gaining obstacle-avoidance support at all (see mpc_controller.py's
_obstacle_constraints -- a linearized/convex approximation of the same
keep-out constraint NMPC enforces exactly).

Saves ../results/comparative_study.json (full results) and
../results/comparative_study_summary.csv (flattened table) plus feeds
research_plots.py for the publication-style figures.
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import torch

import moving_obstacle_demo as scenario_a
import multimodal_obstacle_demo as scenario_b
from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from mpc_controller import MPCConfig, MPCController
from nmpc_controller import NMPCConfig, NMPCController
from ssm_predictor import MultimodalObstaclePredictor, ObstaclePredictor
from train_multimodal_predictor import MODEL_KWARGS as MM_MODEL_KWARGS
from train_predictor import MODEL_KWARGS as UNI_MODEL_KWARGS
from trajectory import cumulative_arclength, nearest_index, reference_horizon
from trajectory_baselines import predict_cv, predict_ctrv
from vehicle_model import KinematicBicycleModel

CONTROLLERS = ["kinematic_ltvmpc", "dynamic_nmpc"]
METHODS = ["naive", "cv", "ctrv", "ssm_uni", "ssm_mm"]
SCENARIOS = ["moving_stop", "ambiguous_go", "ambiguous_stop"]

K, H, DT = scenario_a.K, scenario_a.H, scenario_a.DT
OBSTACLE_RADIUS = scenario_a.OBSTACLE_RADIUS
EGO_SAFETY_MARGIN = scenario_a.EGO_SAFETY_MARGIN
SENSOR_NOISE_STD = scenario_a.SENSOR_NOISE_STD
V_TARGET = scenario_a.V_TARGET


def lane_path():
    return scenario_a.lane_path()


def true_trajectory(scenario: str, sim_steps: int, rng: np.random.Generator) -> np.ndarray:
    if scenario == "moving_stop":
        return scenario_a.pedestrian_true_trajectory(sim_steps, rng)
    elif scenario == "ambiguous_go":
        return scenario_b.pedestrian_true_trajectory("go", sim_steps, rng)
    elif scenario == "ambiguous_stop":
        return scenario_b.pedestrian_true_trajectory("stop", sim_steps, rng)
    raise ValueError(scenario)


def predict_hypotheses(method: str, models: dict, obs_hist_positions: np.ndarray):
    """Returns a list of (H+1, 2) predicted-position arrays -- one entry for
    naive/cv/ctrv/ssm_uni, TWO for ssm_mm (scenario-based avoidance)."""
    now = obs_hist_positions[-1]
    past_disp = np.diff(obs_hist_positions, axis=0)

    if method == "naive":
        offsets = np.zeros((H, 2))
        return [np.vstack([now, now + offsets])]
    elif method == "cv":
        offsets = predict_cv(past_disp, H)
        return [np.vstack([now, now + offsets])]
    elif method == "ctrv":
        offsets = predict_ctrv(past_disp, H, DT)
        return [np.vstack([now, now + offsets])]
    elif method == "ssm_uni":
        with torch.no_grad():
            past_t = torch.tensor(past_disp, dtype=torch.float32).unsqueeze(0)
            pred_offsets, _ = models["ssm_uni"](past_t)
        offsets = pred_offsets.squeeze(0).numpy()
        return [np.vstack([now, now + offsets])]
    elif method == "ssm_mm":
        with torch.no_grad():
            past_t = torch.tensor(past_disp, dtype=torch.float32).unsqueeze(0)
            pred_offsets, _, _ = models["ssm_mm"](past_t)
        pred_offsets = pred_offsets.squeeze(0).numpy()
        return [np.vstack([now, now + pred_offsets[m]]) for m in range(pred_offsets.shape[0])]
    else:
        raise ValueError(method)


def build_controller(controller_name: str):
    if controller_name == "kinematic_ltvmpc":
        model = KinematicBicycleModel(dt=DT)
        controller = MPCController(model, MPCConfig(horizon=H))
        ego_state = np.array([0.0, 0.0, 0.0, V_TARGET * 0.6])
    elif controller_name == "dynamic_nmpc":
        model = DynamicBicycleModel(params=VehicleParams(), dt=DT)
        controller = NMPCController(model, NMPCConfig(horizon=H))
        ego_state = np.array([0.0, 0.0, 0.0, V_TARGET * 0.6, 0.0, 0.0])
    else:
        raise ValueError(controller_name)
    return model, controller, ego_state


def run_scenario(controller_name: str, method: str, models: dict, ped_traj: np.ndarray,
                  sim_steps: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    path = lane_path()
    s = cumulative_arclength(path)
    ego_model, controller, ego_state = build_controller(controller_name)

    ego_states = [ego_state.copy()]
    min_dist = np.inf
    prev_idx = None
    solve_times = []

    for step in range(sim_steps):
        ped_now_idx = min(step, len(ped_traj) - 1)
        window_start = max(0, ped_now_idx - K)
        true_window = ped_traj[window_start: ped_now_idx + 1]
        if len(true_window) < K + 1:
            true_window = np.vstack([np.tile(ped_traj[0], (K + 1 - len(true_window), 1)), true_window])
        noisy_window = true_window + rng.normal(0, SENSOR_NOISE_STD, size=true_window.shape)

        hypotheses = predict_hypotheses(method, models, noisy_window)

        idx = nearest_index(path, ego_state[:2], prev_idx=prev_idx)
        prev_idx = idx
        ref_h = reference_horizon(path, s, idx, H, DT, v_ref=path[idx, 3])

        obstacle_spec = [(hyp, OBSTACLE_RADIUS + EGO_SAFETY_MARGIN) for hyp in hypotheses]
        t0 = time.perf_counter()
        u = controller.solve(ego_state, ref_h, obstacles=obstacle_spec)
        solve_times.append(time.perf_counter() - t0)
        ego_state = ego_model.step(ego_state, u)
        ego_states.append(ego_state.copy())

        true_ped_pos = ped_traj[min(step + 1, len(ped_traj) - 1)]
        dist = np.hypot(ego_state[0] - true_ped_pos[0], ego_state[1] - true_ped_pos[1])
        min_dist = min(min_dist, dist)

        if idx >= len(path) - 5:
            break

    states = np.array(ego_states)
    return {
        "states": states,
        "min_dist": float(min_dist),
        "collision": bool(min_dist < OBSTACLE_RADIUS),
        "max_lateral_deviation": float(np.max(np.abs(states[:, 1]))),
        "mean_solve_time_ms": float(np.mean(solve_times) * 1000),
        "max_solve_time_ms": float(np.max(solve_times) * 1000),
    }


def run_study(n_trials: int = 4, sim_steps: int = 110, verbose: bool = True) -> dict:
    ssm_uni = ObstaclePredictor(**UNI_MODEL_KWARGS)
    ssm_uni.load_state_dict(torch.load("../results/ssm_predictor.pt"))
    ssm_uni.eval()
    ssm_mm = MultimodalObstaclePredictor(**MM_MODEL_KWARGS)
    ssm_mm.load_state_dict(torch.load("../results/ssm_predictor_multimodal.pt"))
    ssm_mm.eval()
    models = {"ssm_uni": ssm_uni, "ssm_mm": ssm_mm}

    results = {}
    first_trial = {}
    for scenario in SCENARIOS:
        results[scenario] = {}
        for controller_name in CONTROLLERS:
            results[scenario][controller_name] = {}
            for method in METHODS:
                metrics = {"min_dist": [], "max_lateral_deviation": [], "collision": [],
                           "mean_solve_time_ms": [], "max_solve_time_ms": []}
                for trial in range(n_trials):
                    seed = 1000 * SCENARIOS.index(scenario) + 7 + trial
                    ped_rng = np.random.default_rng(seed)
                    ped_traj = true_trajectory(scenario, sim_steps + K + 2, ped_rng)[: sim_steps + 2]
                    r = run_scenario(controller_name, method, models, ped_traj, sim_steps, seed=2000 + seed)
                    for k in metrics:
                        metrics[k].append(r[k])
                    if trial == 0:
                        first_trial[(scenario, controller_name, method)] = (ped_traj, r)
                    if verbose:
                        print(f"{scenario:14s} {controller_name:16s} {method:8s} trial {trial}  "
                              f"closest={r['min_dist']:.2f}m  lat_dev={r['max_lateral_deviation']:.2f}m  "
                              f"solve={r['mean_solve_time_ms']:.1f}ms" + ("  COLLISION" if r["collision"] else ""))
                results[scenario][controller_name][method] = {
                    "min_dist_mean": float(np.mean(metrics["min_dist"])),
                    "min_dist_std": float(np.std(metrics["min_dist"])),
                    "max_lateral_deviation_mean": float(np.mean(metrics["max_lateral_deviation"])),
                    "max_lateral_deviation_std": float(np.std(metrics["max_lateral_deviation"])),
                    "mean_solve_time_ms": float(np.mean(metrics["mean_solve_time_ms"])),
                    "max_solve_time_ms": float(np.max(metrics["max_solve_time_ms"])),
                    "n_collisions": int(sum(metrics["collision"])),
                    "n_trials": n_trials,
                }
    return results, first_trial


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    t0 = time.time()
    results, first_trial = run_study(n_trials=4, sim_steps=110, verbose=True)
    print(f"\nTotal wall time: {time.time() - t0:.1f}s")

    print("\n=== Summary: closest approach (m), mean +/- std, collisions/trials ===")
    for scenario in SCENARIOS:
        print(f"\n--- {scenario} ---")
        print(f"{'controller':16s} {'method':8s} {'closest':>16s} {'lat.dev':>16s} {'solve(ms)':>10s} {'collisions':>11s}")
        for controller_name in CONTROLLERS:
            for method in METHODS:
                s = results[scenario][controller_name][method]
                print(f"{controller_name:16s} {method:8s} "
                      f"{s['min_dist_mean']:6.2f}+/-{s['min_dist_std']:<5.2f} "
                      f"{s['max_lateral_deviation_mean']:6.2f}+/-{s['max_lateral_deviation_std']:<5.2f} "
                      f"{s['mean_solve_time_ms']:10.1f} "
                      f"{s['n_collisions']:8d}/{s['n_trials']}")

    with open("../results/comparative_study.json", "w") as f:
        json.dump(results, f, indent=2)

    # Save first-trial trajectories for plotting (only picklable numeric data)
    plot_data = {}
    for (scenario, controller_name, method), (ped_traj, r) in first_trial.items():
        key = f"{scenario}|{controller_name}|{method}"
        plot_data[key] = {"ped_traj": ped_traj.tolist(), "states": r["states"].tolist()}
    with open("../results/comparative_study_trials.json", "w") as f:
        json.dump(plot_data, f)

    print("\nSaved ../results/comparative_study.json and ../results/comparative_study_trials.json")
