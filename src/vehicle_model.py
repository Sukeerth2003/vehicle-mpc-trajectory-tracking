"""
Kinematic bicycle model of a ground vehicle.

State:      x = [X, Y, psi, v]^T
              X, Y  - position of the rear axle (or CoG) in the world frame [m]
              psi   - heading angle [rad]
              v     - longitudinal speed [m/s]

Control:    u = [a, delta]^T
              a      - longitudinal acceleration [m/s^2]
              delta  - front steering angle [rad]

Continuous-time dynamics (rear-axle reference point):

    Xdot   = v * cos(psi)
    Ydot   = v * sin(psi)
    psidot = v / L * tan(delta)
    vdot   = a

where L is the wheelbase.

This model is nonlinear, so for MPC we linearize it around a reference
(operating) point at every control step -> Linear Time-Varying MPC (LTV-MPC).
This is the standard trick used to keep the online optimization a convex QP
while still tracking a curved/varying-speed reference reasonably well.

A note on heading wraparound: psi is deliberately kept as an unbounded
(continuous) real number rather than wrapped into [-pi, pi] after every
step. cos()/sin() make the *dynamics* invariant to that choice, but the
*cost function* used by the MPC (a plain quadratic (psi - psi_ref)^2) is
not invariant to it: if psi were wrapped, the reference heading could sit
at, say, -3.10 rad while the (also-wrapped) vehicle heading sits at
+3.10 rad -- physically 0.08 rad apart, but ~6.2 rad apart numerically,
which briefly convinces the controller a huge correction is needed. Trajectory
headings (see trajectory.py) are generated the same way, via np.unwrap, so
both sides of every error term stay on a consistent, continuous branch.
"""

from __future__ import annotations

import numpy as np

N_STATES = 4  # [X, Y, psi, v]
N_CONTROLS = 2  # [a, delta]


class KinematicBicycleModel:
    """Kinematic bicycle model with RK4 discretization and analytic linearization."""

    def __init__(self, wheelbase: float = 2.7, dt: float = 0.1,
                 max_speed: float = 15.0, min_speed: float = -3.0,
                 max_accel: float = 2.5, max_steer: float = np.deg2rad(35.0)):
        self.L = wheelbase
        self.dt = dt
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_accel = max_accel
        self.max_steer = max_steer

    # ------------------------------------------------------------------
    # Continuous-time dynamics
    # ------------------------------------------------------------------
    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        _, _, psi, v = state
        a, delta = control
        return np.array([
            v * np.cos(psi),
            v * np.sin(psi),
            v / self.L * np.tan(delta),
            a,
        ])

    # ------------------------------------------------------------------
    # Discrete-time step (RK4) -- used to propagate the *true* vehicle
    # ------------------------------------------------------------------
    def step(self, state: np.ndarray, control: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        control = self._clip_control(control)

        k1 = self.continuous_dynamics(state, control)
        k2 = self.continuous_dynamics(state + dt / 2 * k1, control)
        k3 = self.continuous_dynamics(state + dt / 2 * k2, control)
        k4 = self.continuous_dynamics(state + dt * k3, control)
        next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        next_state[3] = np.clip(next_state[3], self.min_speed, self.max_speed)
        # NOTE: heading psi is intentionally left unbounded (not wrapped into
        # [-pi, pi]) -- see the module-level note below for why.
        return next_state

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        a, delta = control
        return np.array([
            np.clip(a, -self.max_accel, self.max_accel),
            np.clip(delta, -self.max_steer, self.max_steer),
        ])

    # ------------------------------------------------------------------
    # Linearization about an operating point (x_bar, u_bar)
    #
    # Continuous-time Jacobians:
    #   A_c = d(f)/d(x),  B_c = d(f)/d(u)
    #
    # Then discretized with a simple forward-Euler (first-order) hold,
    # which is standard practice for LTV-MPC with small dt:
    #   A_d = I + dt * A_c
    #   B_d = dt * B_c
    #   C_d = x_dot(x_bar, u_bar) - A_c @ x_bar - B_c @ u_bar   (affine term,
    #         since f is nonlinear the linearization is only exact locally)
    # ------------------------------------------------------------------
    def linearize(self, x_bar: np.ndarray, u_bar: np.ndarray, dt: float | None = None):
        dt = self.dt if dt is None else dt
        _, _, psi, v = x_bar
        a, delta = u_bar

        A_c = np.zeros((N_STATES, N_STATES))
        A_c[0, 2] = -v * np.sin(psi)
        A_c[0, 3] = np.cos(psi)
        A_c[1, 2] = v * np.cos(psi)
        A_c[1, 3] = np.sin(psi)
        A_c[2, 3] = np.tan(delta) / self.L

        B_c = np.zeros((N_STATES, N_CONTROLS))
        B_c[2, 1] = v / (self.L * np.cos(delta) ** 2)
        B_c[3, 0] = 1.0

        f_bar = self.continuous_dynamics(x_bar, u_bar)

        A_d = np.eye(N_STATES) + dt * A_c
        B_d = dt * B_c
        C_d = dt * (f_bar - A_c @ x_bar - B_c @ u_bar)

        return A_d, B_d, C_d
