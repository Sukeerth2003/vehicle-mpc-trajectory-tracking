"""
Reference trajectory generators for vehicle path tracking.

Each function returns a dense reference path as an (N, 4) array of
[X, Y, psi, v] samples, plus the arc-length spacing used, so the controller
can look up the nearest point and build a local reference horizon.
"""

from __future__ import annotations

import numpy as np


def _heading_and_speed_from_xy(X: np.ndarray, Y: np.ndarray, v_target: float) -> np.ndarray:
    """Given XY samples, back out heading (psi) via finite differences and
    assign a constant target speed. psi is np.unwrap'd so it stays a
    continuous real number with no +-pi discontinuity anywhere along the
    path -- see the note in vehicle_model.py for why that matters for MPC."""
    dX = np.gradient(X)
    dY = np.gradient(Y)
    psi = np.unwrap(np.arctan2(dY, dX))
    v = np.full_like(X, v_target)
    return np.column_stack([X, Y, psi, v])


def figure_eight(v_target: float = 5.0, scale: float = 30.0, n_points: int = 2000) -> np.ndarray:
    """Lemniscate-style figure-eight path."""
    t = np.linspace(0, 2 * np.pi, n_points)
    X = scale * np.sin(t)
    Y = scale * np.sin(t) * np.cos(t)
    return _heading_and_speed_from_xy(X, Y, v_target)


def double_lane_change(v_target: float = 8.0, n_points: int = 2000) -> np.ndarray:
    """ISO-3888-style double lane change maneuver."""
    X = np.linspace(0, 140, n_points)
    Y = np.zeros_like(X)

    def smoothstep(a, b, x):
        t = np.clip((x - a) / (b - a), 0.0, 1.0)
        return t * t * (3 - 2 * t)

    lane_shift = 3.5
    Y += lane_shift * smoothstep(30, 55, X)
    Y -= lane_shift * smoothstep(75, 100, X)
    return _heading_and_speed_from_xy(X, Y, v_target)


def circular(v_target: float = 5.0, radius: float = 15.0, n_points: int = 2000) -> np.ndarray:
    """Constant-radius circular path (good for steady-state cornering analysis)."""
    t = np.linspace(0, 2 * np.pi, n_points)
    X = radius * np.cos(t)
    Y = radius * np.sin(t)
    return _heading_and_speed_from_xy(X, Y, v_target)


TRAJECTORIES = {
    "figure_eight": figure_eight,
    "double_lane_change": double_lane_change,
    "circular": circular,
}


def get_trajectory(name: str, **kwargs) -> np.ndarray:
    if name not in TRAJECTORIES:
        raise ValueError(f"Unknown trajectory '{name}'. Choose from {list(TRAJECTORIES)}")
    return TRAJECTORIES[name](**kwargs)


def nearest_index(path: np.ndarray, position: np.ndarray, prev_idx: int | None = None,
                   window: int = 150) -> int:
    """Index of the closest path point to `position` = [X, Y].

    When `prev_idx` is given, the search is restricted to a window around it
    (mostly looking forward, a little backward). This matters for
    self-intersecting paths like a figure-eight: a *global* nearest-point
    search can suddenly jump to the other lobe of the path when the vehicle
    passes near the crossing point, which yanks the reference heading/curvature
    to a completely different part of the trajectory and destabilizes the
    controller. Restricting the search to "roughly where we already were"
    keeps tracking continuous along the direction of travel.
    """
    if prev_idx is None:
        d2 = (path[:, 0] - position[0]) ** 2 + (path[:, 1] - position[1]) ** 2
        return int(np.argmin(d2))

    lo = max(0, prev_idx - window // 4)
    hi = min(len(path), prev_idx + window)
    seg = path[lo:hi]
    d2 = (seg[:, 0] - position[0]) ** 2 + (seg[:, 1] - position[1]) ** 2
    return lo + int(np.argmin(d2))


def cumulative_arclength(path: np.ndarray) -> np.ndarray:
    """Cumulative arc-length s(i) along the densely-sampled path, s(0) = 0."""
    d = np.hypot(np.diff(path[:, 0]), np.diff(path[:, 1]))
    return np.concatenate([[0.0], np.cumsum(d)])


def reference_horizon(path: np.ndarray, s: np.ndarray, start_idx: int, horizon: int,
                       dt: float, v_ref: float | None = None, min_lookahead_speed: float = 2.0) -> np.ndarray:
    """Pull `horizon` reference points ahead of `start_idx`, spaced by the
    *distance* the vehicle would actually travel in `dt` at `v_ref`
    (arc-length lookup), not by a fixed number of raw path samples --
    otherwise the horizon length in meters silently depends on how densely
    the path was sampled. Clamped at the end of the path (no wraparound)."""
    if v_ref is None:
        v_ref = path[start_idx, 3]
    v_ref = max(v_ref, min_lookahead_speed)
    target_s = s[start_idx] + v_ref * dt * np.arange(0, horizon + 1)
    target_s = np.clip(target_s, 0.0, s[-1])
    idxs = np.searchsorted(s, target_s)
    idxs = np.clip(idxs, 0, len(path) - 1)
    return path[idxs]
