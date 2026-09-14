"""
Linear Time-Varying MPC (LTV-MPC) for trajectory tracking with the
kinematic bicycle model.

At every control step we:
  1. Take a horizon of reference points [X_ref, Y_ref, psi_ref, v_ref].
  2. Estimate the reference curvature -> a nominal steering angle
     delta_ref = atan(L * kappa), and a_ref = 0 (constant reference speed
     along the horizon in this project).
  3. Linearize the kinematic bicycle model about (x_ref_k, u_ref_k) at each
     step of the horizon -> time-varying (A_k, B_k, C_k).
  4. Build and solve a convex QP:

        min  sum_k (x_k - x_ref_k)' Q (x_k - x_ref_k)
                  + u_k' R u_k
                  + (u_k - u_{k-1})' Rd (u_k - u_{k-1})
             + (x_N - x_ref_N)' Qf (x_N - x_ref_N)

        s.t. x_{k+1} = A_k x_k + B_k u_k + C_k
             x_0 = current state
             u_min <= u_k <= u_max
             |u_k - u_{k-1}| <= du_max

  5. Apply only the first control input (receding horizon), then re-solve
     next step with the updated true state -> classic MPC.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cvxpy as cp
import numpy as np

from vehicle_model import KinematicBicycleModel, N_CONTROLS, N_STATES


@dataclass
class MPCConfig:
    horizon: int = 12
    Q: np.ndarray = field(default_factory=lambda: np.diag([15.0, 15.0, 3.0, 2.0]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([25.0, 25.0, 5.0, 2.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 20.0]))
    Rd: np.ndarray = field(default_factory=lambda: np.diag([5.0, 40.0]))
    max_steer_rate: float = np.deg2rad(60.0)   # rad/s
    max_accel_rate: float = 4.0                # m/s^3


class MPCController:
    def __init__(self, model: KinematicBicycleModel, config: MPCConfig | None = None):
        self.model = model
        self.cfg = config or MPCConfig()
        self.prev_u = np.zeros(N_CONTROLS)
        self.last_predicted_states: np.ndarray | None = None  # for visualization

    # ------------------------------------------------------------------
    def _estimate_reference_controls(self, ref_horizon: np.ndarray) -> np.ndarray:
        """Estimate a feasible nominal control sequence along the reference
        horizon from its geometry: delta_ref from curvature, a_ref = 0."""
        psi = ref_horizon[:, 2]
        v = ref_horizon[:, 3]
        dpsi = np.diff(psi)
        dpsi = np.arctan2(np.sin(dpsi), np.cos(dpsi))  # wrap
        ds = np.maximum(v[:-1] * self.model.dt, 1e-3)
        kappa = dpsi / ds
        delta_ref = np.arctan(self.model.L * kappa)
        delta_ref = np.clip(delta_ref, -self.model.max_steer, self.model.max_steer)
        delta_ref = np.append(delta_ref, delta_ref[-1] if len(delta_ref) else 0.0)
        a_ref = np.zeros_like(delta_ref)
        return np.column_stack([a_ref, delta_ref])

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, ref_horizon: np.ndarray) -> np.ndarray:
        """ref_horizon: (H+1, 4) array of [X,Y,psi,v] reference points.
        Returns the first control action u* = [a, delta]."""
        H = self.cfg.horizon
        assert ref_horizon.shape[0] >= H + 1, "reference horizon too short"

        u_ref_seq = self._estimate_reference_controls(ref_horizon)

        # Build time-varying linearization along the reference horizon
        A_list, B_list, C_list = [], [], []
        for k in range(H):
            A_k, B_k, C_k = self.model.linearize(ref_horizon[k], u_ref_seq[k])
            A_list.append(A_k)
            B_list.append(B_k)
            C_list.append(C_k)

        x = cp.Variable((N_STATES, H + 1))
        u = cp.Variable((N_CONTROLS, H))

        cost = 0
        constraints = [x[:, 0] == x0]

        for k in range(H):
            x_ref_k = ref_horizon[k]
            e = x[:, k] - x_ref_k
            # wrap heading error into cost via linear approx (kept simple: use raw diff,
            # reference psi is unwrapped-consistent because trajectories are smooth)
            cost += cp.quad_form(e, self.cfg.Q)
            cost += cp.quad_form(u[:, k], self.cfg.R)

            u_prev = self.prev_u if k == 0 else u[:, k - 1]
            cost += cp.quad_form(u[:, k] - u_prev, self.cfg.Rd)

            constraints += [
                x[:, k + 1] == A_list[k] @ x[:, k] + B_list[k] @ u[:, k] + C_list[k],
                u[0, k] >= -self.model.max_accel,
                u[0, k] <= self.model.max_accel,
                u[1, k] >= -self.model.max_steer,
                u[1, k] <= self.model.max_steer,
            ]
            if k > 0:
                constraints += [
                    cp.abs(u[0, k] - u[0, k - 1]) <= self.cfg.max_accel_rate * self.model.dt,
                    cp.abs(u[1, k] - u[1, k - 1]) <= self.cfg.max_steer_rate * self.model.dt,
                ]

        e_terminal = x[:, H] - ref_horizon[H]
        cost += cp.quad_form(e_terminal, self.cfg.Qf)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            problem.solve(solver=cp.ECOS, verbose=False)

        if u.value is None or x.value is None:
            # Solver failed (infeasible / numerical issue): fall back to holding
            # the previous command rather than crashing the simulation.
            self.last_predicted_states = np.tile(x0, (H + 1, 1)).T
            return self.prev_u.copy()

        self.last_predicted_states = x.value
        u_opt = u.value[:, 0]
        self.prev_u = u_opt.copy()
        return u_opt
