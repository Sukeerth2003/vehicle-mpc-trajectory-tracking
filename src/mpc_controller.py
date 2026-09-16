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
        self._prev_solution: tuple | None = None  # (X, U) from the last successful solve, for fallback

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
    def _obstacle_constraints(self, x, ref_horizon, obstacles):
        """Convex, linearized keep-out constraints for the QP -- the LTV-MPC
        analog of nmpc_controller's hard nonconvex ||pos - center|| >= radius
        constraint (see the README for the honest comparison between the
        two). A QP can't express a nonconvex disk exclusion directly, so
        each obstacle's keep-out circle is replaced with its *supporting
        half-plane* at the point on the circle nearest a linearization
        point: n = (lin_point - obs_center) / ||...||, constraint
        n . (pos - obs_center) >= radius. That's exact exactly on the
        tangent line through lin_point's nearest approach and increasingly
        conservative (or, in principle, permissive on the wrong side) the
        further the true optimal trajectory ends up from lin_point -- unlike
        NMPC's per-solve exact nonconvex constraint, this is only as good as
        the linearization point. lin_point is taken from the PREVIOUS solve's
        predicted position at that horizon step when available (the same
        "trust last step's plan" idea used for the failure fallback below),
        falling back to the reference path on the very first solve.
        `obstacles` uses the exact same `(center, radius)` format as
        nmpc_controller.solve (static (2,) or moving (H+1, 2) predicted
        centers, k=0..H) so the same obstacle-hypothesis code feeds both
        controllers in the comparison study.
        Returns a list of cvxpy constraints (possibly empty)."""
        if not obstacles:
            return []
        H = self.cfg.horizon
        cons = []
        prev_X = self._prev_solution[0] if self._prev_solution is not None else None
        for entry in obstacles:
            center, radius = (entry[:2], entry[2]) if len(entry) == 3 else entry
            center = np.asarray(center, dtype=float)
            centers_k = np.tile(center, (H + 1, 1)) if center.ndim == 1 else center
            for k in range(H + 1):
                obs_k = centers_k[k]
                lin_point = prev_X[:2, k] if (prev_X is not None and prev_X.shape[1] > k) else ref_horizon[k, :2]
                d = lin_point - obs_k
                dist = np.linalg.norm(d)
                n = d / dist if dist > 1e-6 else np.array([0.0, 1.0])
                cons.append(n @ (x[:2, k] - obs_k) >= radius)
        return cons

    # ------------------------------------------------------------------
    def solve(self, x0: np.ndarray, ref_horizon: np.ndarray,
              obstacles: list[tuple] | None = None) -> np.ndarray:
        """ref_horizon: (H+1, 4) array of [X,Y,psi,v] reference points.
        obstacles: optional list of `(center, radius)` keep-out zones, same
        format nmpc_controller.solve accepts -- see _obstacle_constraints
        for how a QP handles these (linearized, not exact).
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

        constraints += self._obstacle_constraints(x, ref_horizon, obstacles)

        problem = cp.Problem(cp.Minimize(cost), constraints)
        try:
            problem.solve(solver=cp.OSQP, warm_start=True, verbose=False)
        except cp.SolverError:
            try:
                problem.solve(solver=cp.ECOS, verbose=False)
            except cp.SolverError:
                pass

        if u.value is None or x.value is None:
            # Solver failed (infeasible -- most often the linearized obstacle
            # half-plane conflicting with actuator limits -- or a numerical
            # issue). When an obstacle is active, reuse the PREVIOUS solve's
            # shifted trajectory rather than blindly holding the previous
            # command: the same "obstacle-blind fallback causes real
            # collisions" bug documented for NMPC (see the README) applies
            # here just as much -- holding the last command has no notion of
            # the obstacle either. Only fall back to holding the command when
            # there is no obstacle active or no previous solution to reuse.
            if obstacles and self._prev_solution is not None:
                X_opt, U_opt = self._prev_solution
                self.last_predicted_states = X_opt
                u_opt = U_opt[:, 0]
                self.prev_u = u_opt.copy()
                self._prev_solution = (np.hstack([X_opt[:, 1:], X_opt[:, -1:]]),
                                        np.hstack([U_opt[:, 1:], U_opt[:, -1:]]))
                return u_opt
            self.last_predicted_states = np.tile(x0, (H + 1, 1)).T
            return self.prev_u.copy()

        self.last_predicted_states = x.value
        u_opt = u.value[:, 0]
        self.prev_u = u_opt.copy()
        X_shifted = np.hstack([x.value[:, 1:], x.value[:, -1:]])
        U_shifted = np.hstack([u.value[:, 1:], u.value[:, -1:]])
        self._prev_solution = (X_shifted, U_shifted)
        return u_opt
