"""
Multimodal, scenario-based moving-obstacle avoidance demo -- the payoff for
train_multimodal_predictor.py.

Same lane-crossing-pedestrian scenario as moving_obstacle_demo.py, but this
time the pedestrian's true behavior is GENUINELY AMBIGUOUS during the ego's
approach: on any given trial, the pedestrian either keeps walking straight
through the lane ("go") or decelerates to a stop right in the ego's path
("stop") -- decided by a coin flip that the ego cannot observe in advance
(see obstacle_trajectory_data.py's "branch" pattern, which this scenario's
true trajectory generator deliberately mirrors). This is the case Part 3's
predictors -- CV and the unimodal SSM -- are fundamentally unable to handle
correctly: both are built to output ONE trajectory, so the best either can
do on a genuinely 50/50 case is predict something *between* "go" and "stop",
which matches neither outcome.

Three prediction strategies feed the exact same NMPC controller
(nmpc_controller.py), on the exact same true pedestrian trajectory and
sensor noise realization per trial:

  - cv        -- constant-velocity extrapolation (trajectory_baselines),
                 ONE hypothesis, fed as one obstacle constraint slot.
  - ssm_uni   -- Part 3's unimodal ObstaclePredictor, ONE hypothesis, one
                 constraint slot.
  - ssm_mm    -- train_multimodal_predictor.py's MultimodalObstaclePredictor,
                 TWO hypotheses (mode 0: "keeps moving", mode 1: "slows
                 down"), fed as TWO SEPARATE hard constraint slots -- the
                 controller must stay clear of BOTH simultaneously
                 (scenario-based avoidance; see nmpc_controller.solve's
                 docstring), not just whichever one is "more likely".

Trials are run with the true branch FORCED (not left to a coin flip) so
"go" and "stop" trials are equally represented and every method sees the
exact same set of true trajectories -- a fair, matched comparison, in the
same spirit as robustness_experiment.py and moving_obstacle_demo.py.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from nmpc_controller import NMPCConfig, NMPCController
from ssm_predictor import MultimodalObstaclePredictor, ObstaclePredictor
from train_multimodal_predictor import MODEL_KWARGS as MM_MODEL_KWARGS
from train_predictor import MODEL_KWARGS as UNI_MODEL_KWARGS
from trajectory import cumulative_arclength, nearest_index, reference_horizon
from trajectory_baselines import predict_cv

DT = 0.1
V_TARGET = 8.0
LANE_LENGTH = 100.0
K, H = 10, 10
OBSTACLE_RADIUS = 0.6
SENSOR_NOISE_STD = 0.05
PED_CROSS_X = 45.0
PED_START_Y = -6.0
EGO_SAFETY_MARGIN = 1.0


def lane_path(n_points=2000):
    X = np.linspace(0, LANE_LENGTH, n_points)
    Y = np.zeros_like(X)
    psi = np.zeros_like(X)
    v = np.full_like(X, V_TARGET)
    return np.column_stack([X, Y, psi, v])


def pedestrian_true_trajectory(branch: str, n_total_steps: int, rng: np.random.Generator) -> np.ndarray:
    """`branch` is forced ("go" or "stop"), not randomized here -- see the
    module docstring for why. "stop" reuses the same analytic
    stopping-point logic moving_obstacle_demo.py uses (deceleration onset
    computed from the randomized speed/decel draw so it reliably stops IN
    the lane); "go" just keeps walking through at ~constant speed. Returns
    (n_total_steps, 2) world [x, y] positions."""
    v = rng.uniform(1.2, 1.6)
    x, y = PED_CROSS_X, PED_START_Y
    xs, ys = [x], [y]

    if branch == "go":
        for _ in range(n_total_steps - 1):
            y += v * DT
            xs.append(x); ys.append(y)
        return np.column_stack([xs, ys])

    decel = rng.uniform(0.9, 1.3)
    target_stop_y = rng.uniform(-0.3, 0.3)
    stopping_distance = v ** 2 / (2 * decel)
    decel_start_y = target_stop_y - stopping_distance
    for _ in range(n_total_steps - 1):
        if y >= decel_start_y:
            v = max(0.0, v - decel * DT)
        y += v * DT
        xs.append(x); ys.append(y)
    return np.column_stack([xs, ys])


def _predict_obstacle_slots(method: str, models: dict, obs_hist_positions: np.ndarray):
    """obs_hist_positions: (K+1, 2) recent (noisy) observed positions, most
    recent last. Returns a list of (H+1, 2) predicted-position arrays -- ONE
    entry for cv/ssm_uni, TWO for ssm_mm (one per mode)."""
    now = obs_hist_positions[-1]
    past_disp = np.diff(obs_hist_positions, axis=0)   # (K, 2)

    if method == "cv":
        offsets = predict_cv(past_disp, H)
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
            pred_offsets, _, _ = models["ssm_mm"](past_t)   # (1, M, H, 2)
        pred_offsets = pred_offsets.squeeze(0).numpy()       # (M, H, 2)
        return [np.vstack([now, now + pred_offsets[m]]) for m in range(pred_offsets.shape[0])]

    else:
        raise ValueError(method)


def run_scenario(method: str, models: dict, branch: str, ped_traj: np.ndarray, sim_steps: int, seed: int):
    rng = np.random.default_rng(seed)
    path = lane_path()
    s = cumulative_arclength(path)
    ego_model = DynamicBicycleModel(params=VehicleParams(), dt=DT)
    controller = NMPCController(ego_model, NMPCConfig(horizon=H))

    ego_state = np.array([0.0, 0.0, 0.0, V_TARGET * 0.6, 0.0, 0.0])
    ego_states = [ego_state.copy()]
    min_dist = np.inf
    prev_idx = None

    for step in range(sim_steps):
        ped_now_idx = min(step, len(ped_traj) - 1)
        window_start = max(0, ped_now_idx - K)
        true_window = ped_traj[window_start: ped_now_idx + 1]
        if len(true_window) < K + 1:
            true_window = np.vstack([np.tile(ped_traj[0], (K + 1 - len(true_window), 1)), true_window])
        noisy_window = true_window + rng.normal(0, SENSOR_NOISE_STD, size=true_window.shape)

        hypotheses = _predict_obstacle_slots(method, models, noisy_window)

        idx = nearest_index(path, ego_state[:2], prev_idx=prev_idx)
        prev_idx = idx
        ref_h = reference_horizon(path, s, idx, H, DT, v_ref=path[idx, 3])

        obstacle_spec = [(hyp, OBSTACLE_RADIUS + EGO_SAFETY_MARGIN) for hyp in hypotheses]
        u = controller.solve(ego_state, ref_h, obstacles=obstacle_spec)
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
        "min_dist": min_dist,
        "collision": min_dist < OBSTACLE_RADIUS,
        "max_lateral_deviation": float(np.max(np.abs(states[:, 1]))),
    }


def run_trial(models: dict, branch: str, sim_steps: int, trial_seed: int) -> dict:
    ped_rng = np.random.default_rng(trial_seed)
    ped_traj = pedestrian_true_trajectory(branch, sim_steps + K + 2, ped_rng)[: sim_steps + 2]
    return ped_traj, {method: run_scenario(method, models, branch, ped_traj, sim_steps, seed=1000 + trial_seed)
                       for method in ["cv", "ssm_uni", "ssm_mm"]}


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    ssm_uni = ObstaclePredictor(**UNI_MODEL_KWARGS)
    ssm_uni.load_state_dict(torch.load("../results/ssm_predictor.pt"))
    ssm_uni.eval()
    ssm_mm = MultimodalObstaclePredictor(**MM_MODEL_KWARGS)
    ssm_mm.load_state_dict(torch.load("../results/ssm_predictor_multimodal.pt"))
    ssm_mm.eval()
    models = {"ssm_uni": ssm_uni, "ssm_mm": ssm_mm}

    sim_steps = 130
    n_trials_per_branch = 6
    methods = ["cv", "ssm_uni", "ssm_mm"]
    all_results = {branch: {m: {"min_dist": [], "max_lateral_deviation": [], "collision": []} for m in methods}
                    for branch in ["go", "stop"]}
    first_trial = {}

    for branch in ["go", "stop"]:
        for trial in range(n_trials_per_branch):
            seed = 7 + trial + (100 if branch == "stop" else 0)
            ped_traj, results = run_trial(models, branch, sim_steps, trial_seed=seed)
            if trial == 0:
                first_trial[branch] = (ped_traj, results)
            for method in methods:
                r = results[method]
                all_results[branch][method]["min_dist"].append(r["min_dist"])
                all_results[branch][method]["max_lateral_deviation"].append(r["max_lateral_deviation"])
                all_results[branch][method]["collision"].append(r["collision"])
                print(f"branch={branch:4s} trial {trial}  {method:8s}  closest approach = {r['min_dist']:.2f} m  "
                      f"max lateral deviation = {r['max_lateral_deviation']:.2f} m"
                      + ("  COLLISION" if r["collision"] else ""))

    print(f"\n--- Summary over {n_trials_per_branch} trials per branch (mean +/- std) ---")
    summary = {}
    for branch in ["go", "stop"]:
        summary[branch] = {}
        for method in methods:
            d = all_results[branch][method]
            n_collisions = int(sum(bool(c) for c in d["collision"]))
            summary[branch][method] = {
                "min_dist_mean": float(np.mean(d["min_dist"])), "min_dist_std": float(np.std(d["min_dist"])),
                "max_lateral_deviation_mean": float(np.mean(d["max_lateral_deviation"])),
                "max_lateral_deviation_std": float(np.std(d["max_lateral_deviation"])),
                "n_collisions": n_collisions,
            }
            s = summary[branch][method]
            print(f"{branch:4s} {method:8s}  closest approach = {s['min_dist_mean']:.2f} +/- {s['min_dist_std']:.2f} m   "
                  f"max lateral deviation = {s['max_lateral_deviation_mean']:.2f} +/- "
                  f"{s['max_lateral_deviation_std']:.2f} m   collisions = {n_collisions}/{n_trials_per_branch}")

    import json
    with open("../results/multimodal_obstacle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    from visualize import animate_moving_obstacle, multimodal_obstacle_plot

    for branch in ["go", "stop"]:
        ped_traj, results = first_trial[branch]
        multimodal_obstacle_plot(lane_path(), ped_traj, results, OBSTACLE_RADIUS, branch,
                                  f"../results/multimodal_obstacle_plot_{branch}.png")
    animate_moving_obstacle(lane_path(), first_trial["stop"][0], first_trial["stop"][1]["ssm_mm"], OBSTACLE_RADIUS,
                             "../results/multimodal_obstacle_tracking.gif",
                             title="NMPC with multimodal (scenario-based) pedestrian avoidance")
    print("\nSaved multimodal_obstacle_plot_go.png, multimodal_obstacle_plot_stop.png, "
          "multimodal_obstacle_tracking.gif, and multimodal_obstacle_summary.json to ../results/")
