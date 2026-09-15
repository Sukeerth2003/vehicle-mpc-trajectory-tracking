"""
Non-learned baseline predictors for obstacle trajectory forecasting, used to
give the SSM predictor's ADE/FDE numbers an honest comparison point (same
role pure pursuit plays for the MPC controllers elsewhere in this project).

Both operate on the same (past_disp: (K,2)) input as ssm_predictor and
return (H,2) predicted cumulative offsets from the last observed position.
"""

from __future__ import annotations

import numpy as np


def predict_cv(past_disp: np.ndarray, H: int) -> np.ndarray:
    """Constant velocity: repeat the last observed per-step displacement."""
    last = past_disp[-1]
    return np.cumsum(np.tile(last, (H, 1)), axis=0)


def predict_ctrv(past_disp: np.ndarray, H: int, dt: float) -> np.ndarray:
    """Constant turn-rate and velocity: estimate speed and yaw rate from the
    last two observed displacements, then integrate a constant-turn arc
    forward -- the same maneuver model classical trackers (e.g. an
    extended Kalman filter tracking a vehicle) commonly assume."""
    last = past_disp[-1]
    prev = past_disp[-2] if len(past_disp) > 1 else last

    speed = np.linalg.norm(last) / dt
    heading = np.arctan2(last[1], last[0])
    prev_heading = np.arctan2(prev[1], prev[0]) if np.linalg.norm(prev) > 1e-6 else heading
    turn_rate = np.arctan2(np.sin(heading - prev_heading), np.cos(heading - prev_heading)) / dt

    offsets = np.zeros((H, 2))
    x, y, psi = 0.0, 0.0, heading
    for h in range(H):
        psi += turn_rate * dt
        x += speed * np.cos(psi) * dt
        y += speed * np.sin(psi) * dt
        offsets[h] = [x, y]
    return offsets


def ade(pred_offsets: np.ndarray, true_offsets: np.ndarray) -> float:
    """Average displacement error over the horizon (mean Euclidean error)."""
    return float(np.mean(np.linalg.norm(pred_offsets - true_offsets, axis=-1)))


def fde(pred_offsets: np.ndarray, true_offsets: np.ndarray) -> float:
    """Final displacement error (Euclidean error at the last horizon step)."""
    return float(np.linalg.norm(pred_offsets[-1] - true_offsets[-1]))
