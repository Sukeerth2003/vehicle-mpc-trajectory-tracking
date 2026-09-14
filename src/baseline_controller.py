"""
Pure Pursuit + PI speed controller.

Used as a classical, non-optimization-based baseline to compare against the
MPC controller: same vehicle model, same trajectories, same actuator
limits -- only the control law differs.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vehicle_model import KinematicBicycleModel


@dataclass
class PurePursuitConfig:
    lookahead_base: float = 3.0   # minimum lookahead distance [m]
    lookahead_gain: float = 0.6   # extra lookahead per m/s of speed
    kp_speed: float = 1.2
    ki_speed: float = 0.15


class PurePursuitController:
    def __init__(self, model: KinematicBicycleModel, config: PurePursuitConfig | None = None):
        self.model = model
        self.cfg = config or PurePursuitConfig()
        self._speed_integral = 0.0
        self.target_point: np.ndarray | None = None  # for visualization

    def reset(self):
        self._speed_integral = 0.0

    def solve(self, x0: np.ndarray, path: np.ndarray, nearest_idx: int) -> np.ndarray:
        X, Y, psi, v = x0
        lookahead = self.cfg.lookahead_base + self.cfg.lookahead_gain * max(v, 0.0)

        # walk forward along the path until we exceed the lookahead distance
        idx = nearest_idx
        while idx < len(path) - 1:
            d = np.hypot(path[idx, 0] - X, path[idx, 1] - Y)
            if d >= lookahead:
                break
            idx += 1
        target = path[idx]
        self.target_point = target[:2]

        # pure pursuit steering law
        dx = target[0] - X
        dy = target[1] - Y
        alpha = np.arctan2(dy, dx) - psi
        alpha = np.arctan2(np.sin(alpha), np.cos(alpha))
        Ld = max(np.hypot(dx, dy), 1e-3)
        delta = np.arctan2(2.0 * self.model.L * np.sin(alpha), Ld)
        delta = np.clip(delta, -self.model.max_steer, self.model.max_steer)

        # PI speed control toward the reference speed at the nearest point
        v_ref = path[nearest_idx, 3]
        err = v_ref - v
        self._speed_integral += err * self.model.dt
        a = self.cfg.kp_speed * err + self.cfg.ki_speed * self._speed_integral
        a = np.clip(a, -self.model.max_accel, self.model.max_accel)

        return np.array([a, delta])
