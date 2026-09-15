"""
Moving-obstacle avoidance demo: the payoff for training ssm_predictor.

Scenario: the ego vehicle (dynamic model + NMPC) drives a straight lane at
cruise speed while a pedestrian starts crossing the lane ahead -- and then
stops partway across (a very ordinary, safety-critical real case: someone
hesitates, or stops to let the car pass). The ego only ever gets a NOISY,
partial view of the pedestrian's recent track (like a real perception
stack), and has to decide, every control step, where the pedestrian will be
over its planning horizon.

Three assumptions about "where will the obstacle be" are compared, feeding
the exact same NMPC moving-obstacle constraint (nmpc_controller.py) with the
exact same true pedestrian trajectory and sensor noise realization:

  - naive/static  -- assume the obstacle stays at its last sensed position
                      for the whole horizon (what obstacle_demo.py effectively
                      does, since it only ever handles truly static obstacles).
  - CV             -- constant-velocity extrapolation (trajectory_baselines).
  - SSM (learned) -- ssm_predictor.ObstaclePredictor's rollout.

The naive assumption is a real, common simplification (it's exactly what you
get if you bolt "static obstacle avoidance" onto a moving world and don't
think any harder about it) and it fails in exactly the way this scenario is
built to expose: once the pedestrian visibly starts decelerating, "naive"
still predicts them crossing straight through and out of the way, so the
constraint stops actively pushing the ego to slow down/steer clear right
when it matters most.
"""

from __future__ import annotations

import os

import numpy as np
import torch

from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from nmpc_controller import NMPCConfig, NMPCController
from ssm_predictor import ObstaclePredictor
from train_predictor import MODEL_KWARGS
from trajectory import cumulative_arclength, nearest_index, reference_horizon
from trajectory_baselines import predict_cv

DT = 0.1
V_TARGET = 8.0
LANE_LENGTH = 100.0
K, H = 10, 10
OBSTACLE_RADIUS = 0.6          # pedestrian-scale
SENSOR_NOISE_STD = 0.05        # meters -- matches train_predictor.py's training noise
PED_CROSS_X = 45.0             # where the pedestrian's path crosses the lane centerline
PED_START_Y = -6.0
EGO_SAFETY_MARGIN = 1.0        # extra clearance folded into the constraint radius


def lane_path(n_points=2000):
    X = np.linspace(0, LANE_LENGTH, n_points)
    Y = np.zeros_like(X)
    psi = np.zeros_like(X)
    v = np.full_like(X, V_TARGET)
    return np.column_stack([X, Y, psi, v])


def pedestrian_true_trajectory(n_total_steps: int, rng: np.random.Generator) -> np.ndarray:
    """The 'decelerating' motion pattern (obstacle_trajectory_data._simulate
    generates the same shape, randomly placed/aimed -- this rebuilds it
    directly so the crossing point and timing are pinned exactly at this
    scenario's PED_CROSS_X / PED_START_Y): walks straight toward the lane
    centerline and decelerates to a stop right around y=0 -- i.e. directly in
    the ego's path, not safely past it. Deceleration onset is computed
    analytically from the (randomized) speed/decel rate so the stopping
    point lands at the target regardless of those draws (an earlier version
    started decelerating at a fixed fraction of the simulation length,
    which -- depending on the randomized speed -- could let the pedestrian
    coast straight through y=0 and stop well clear of the lane on the other
    side, silently turning this into a non-scenario where no obstacle
    avoidance was ever needed; see the README for how that was caught).
    Returns (n_total_steps, 2) world [x, y] positions."""
    decel = rng.uniform(0.9, 1.3)
    v = rng.uniform(1.2, 1.6)
    target_stop_y = rng.uniform(-0.3, 0.3)   # small realistic variation in exactly where they stop
    stopping_distance = v ** 2 / (2 * decel)
    decel_start_y = target_stop_y - stopping_distance

    x, y = PED_CROSS_X, PED_START_Y
    xs, ys = [x], [y]
    for _ in range(n_total_steps - 1):
        if y >= decel_start_y:
            v = max(0.0, v - decel * DT)
        y += v * DT
        xs.append(x); ys.append(y)
    return np.column_stack([xs, ys])


def _predict_positions(method: str, model, obs_hist_positions: np.ndarray) -> np.ndarray:
    """obs_hist_positions: (K+1, 2) recent (noisy) observed obstacle
    positions, most recent last. Returns (H+1, 2) predicted positions
    (index 0 = current position, indices 1..H = predicted future), matching
    what NMPCController.solve's moving-obstacle `center` expects."""
    now = obs_hist_positions[-1]
    past_disp = np.diff(obs_hist_positions, axis=0)   # (K, 2)

    if method == "naive":
        offsets = np.zeros((H, 2))
    elif method == "cv":
        offsets = predict_cv(past_disp, H)
    elif method == "ssm":
        with torch.no_grad():
            past_t = torch.tensor(past_disp, dtype=torch.float32).unsqueeze(0)
            pred_offsets, _ = model(past_t)
        offsets = pred_offsets.squeeze(0).numpy()
    else:
        raise ValueError(method)

    return np.vstack([now, now + offsets])   # (H+1, 2)


def run_scenario(method: str, model, ped_traj: np.ndarray, sim_steps: int, seed: int):
    """Runs the ego NMPC for `sim_steps` control steps against the fixed
    `ped_traj` ground truth, using `method` to predict the pedestrian's
    future path each step. Returns dict of results."""
    rng = np.random.default_rng(seed)
    path = lane_path()
    s = cumulative_arclength(path)
    ego_model = DynamicBicycleModel(params=VehicleParams(), dt=DT)
    controller = NMPCController(ego_model, NMPCConfig(horizon=H))

    ego_state = np.array([0.0, 0.0, 0.0, V_TARGET * 0.6, 0.0, 0.0])   # starts a bit slow, catches up to cruise
    ego_states = [ego_state.copy()]
    min_dist = np.inf
    prev_idx = None

    for step in range(sim_steps):
        ped_now_idx = min(step, len(ped_traj) - 1)
        # K+1 most recent (noisy) sensed pedestrian positions up to "now".
        window_start = max(0, ped_now_idx - K)
        true_window = ped_traj[window_start: ped_now_idx + 1]
        if len(true_window) < K + 1:
            true_window = np.vstack([np.tile(ped_traj[0], (K + 1 - len(true_window), 1)), true_window])
        noisy_window = true_window + rng.normal(0, SENSOR_NOISE_STD, size=true_window.shape)

        pred_positions = _predict_positions(method, model, noisy_window)   # (H+1, 2)

        idx = nearest_index(path, ego_state[:2], prev_idx=prev_idx)
        prev_idx = idx
        ref_h = reference_horizon(path, s, idx, H, DT, v_ref=path[idx, 3])

        obstacle_spec = [(pred_positions, OBSTACLE_RADIUS + EGO_SAFETY_MARGIN)]
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


def run_trial(model, sim_steps: int, trial_seed: int) -> dict:
    """One trial: a fresh randomized pedestrian trajectory + sensor noise
    realization, all three methods run against the identical true path and
    noise (fair, matched comparison -- same pattern robustness_experiment.py
    uses for the controller Monte Carlo study)."""
    ped_rng = np.random.default_rng(trial_seed)
    ped_traj = pedestrian_true_trajectory(sim_steps + K + 2, ped_rng)[: sim_steps + 2]
    return ped_traj, {method: run_scenario(method, model, ped_traj, sim_steps, seed=1000 + trial_seed)
                       for method in ["naive", "cv", "ssm"]}


if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)

    model = ObstaclePredictor(**MODEL_KWARGS)
    model.load_state_dict(torch.load("../results/ssm_predictor.pt"))
    model.eval()

    sim_steps = 130
    n_trials = 8
    all_results = {method: {"min_dist": [], "max_lateral_deviation": [], "collision": []}
                    for method in ["naive", "cv", "ssm"]}
    first_ped_traj, first_results = None, None

    for trial in range(n_trials):
        ped_traj, results = run_trial(model, sim_steps, trial_seed=7 + trial)
        if trial == 0:
            first_ped_traj, first_results = ped_traj, results
        for method in ["naive", "cv", "ssm"]:
            r = results[method]
            all_results[method]["min_dist"].append(r["min_dist"])
            all_results[method]["max_lateral_deviation"].append(r["max_lateral_deviation"])
            all_results[method]["collision"].append(r["collision"])
            print(f"trial {trial}  {method:6s}  closest approach = {r['min_dist']:.2f} m  "
                  f"max lateral deviation = {r['max_lateral_deviation']:.2f} m"
                  + ("  COLLISION" if r["collision"] else ""))

    print(f"\n--- Summary over {n_trials} trials (mean +/- std) ---")
    summary = {}
    for method in ["naive", "cv", "ssm"]:
        d = all_results[method]
        n_collisions = int(sum(bool(c) for c in d["collision"]))
        summary[method] = {
            "min_dist_mean": float(np.mean(d["min_dist"])), "min_dist_std": float(np.std(d["min_dist"])),
            "max_lateral_deviation_mean": float(np.mean(d["max_lateral_deviation"])),
            "max_lateral_deviation_std": float(np.std(d["max_lateral_deviation"])),
            "n_collisions": n_collisions,
        }
        s = summary[method]
        print(f"{method:6s}  closest approach = {s['min_dist_mean']:.2f} +/- {s['min_dist_std']:.2f} m   "
              f"max lateral deviation = {s['max_lateral_deviation_mean']:.2f} +/- "
              f"{s['max_lateral_deviation_std']:.2f} m   collisions = {n_collisions}/{n_trials}")

    import json
    with open("../results/moving_obstacle_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    from visualize import animate_moving_obstacle, moving_obstacle_plot

    moving_obstacle_plot(lane_path(), first_ped_traj, first_results, OBSTACLE_RADIUS,
                          "../results/moving_obstacle_plot.png")
    animate_moving_obstacle(lane_path(), first_ped_traj, first_results["ssm"], OBSTACLE_RADIUS,
                             "../results/moving_obstacle_tracking.gif",
                             title="NMPC with SSM-predicted pedestrian avoidance")
    print("\nSaved moving_obstacle_plot.png, moving_obstacle_tracking.gif, "
          "and moving_obstacle_summary.json to ../results/")
