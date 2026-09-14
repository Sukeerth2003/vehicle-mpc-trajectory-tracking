"""
Dynamic bicycle model with a linear tire model (lateral tire forces from
slip angles), in the vehicle's body-fixed frame.

State:      x = [X, Y, psi, vx, vy, r]^T
              X, Y   - position of the center of gravity (CG) in the world frame [m]
              psi    - heading angle [rad]
              vx     - longitudinal (body-frame) speed [m/s]
              vy     - lateral (body-frame) speed [m/s]
              r      - yaw rate [rad/s]

Control:    u = [ax, delta]^T
              ax     - net longitudinal acceleration [m/s^2] (treated as a direct
                        input, i.e. we do not split it into front/rear drive force --
                        a common simplification when the focus is lateral control)
              delta  - front steering angle [rad]

Why this model, and why it needs NMPC (not just a bigger LTV-MPC)
-------------------------------------------------------------------
The kinematic bicycle model (see vehicle_model.py) assumes zero tire slip: the
wheels always point exactly where the vehicle is actually going. That is a good
approximation at low speed / low lateral acceleration, but it breaks down for
faster or more aggressive maneuvers, where the tires *do* slip sideways relative
to their heading and that slip is what actually generates the lateral force that
turns the car.

This model adds that: front/rear slip angles (alpha_f, alpha_r) computed from the
body-frame velocity components, and a linear tire law Fy = C_alpha * alpha
relating slip angle to lateral tire force. The resulting dynamics are more
nonlinear (through atan2() and products of states) than the kinematic model, and
-- critically -- the useful state now includes vy and r, which don't have a clean
"operating point from the reference path" the way the kinematic model's heading
does. So rather than re-linearize this model every step (LTV-MPC), the NMPC
controller (nmpc_controller.py) solves the true nonlinear dynamics directly via
CasADi + IPOPT.

Reference: R. Rajamani, "Vehicle Dynamics and Control," 2nd ed., Springer, 2011,
Ch. 2 (lateral vehicle dynamics); J. Kong et al., "Kinematic and dynamic vehicle
models for autonomous driving control design," IEEE IV 2015.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

N_STATES = 6   # [X, Y, psi, vx, vy, r]
N_CONTROLS = 2  # [ax, delta]


@dataclass
class VehicleParams:
    mass: float = 1500.0          # kg
    Iz: float = 3000.0            # yaw moment of inertia [kg m^2]
    a: float = 1.2                # CG to front axle [m]
    b: float = 1.5                # CG to rear axle [m]   (wheelbase L = a + b)
    Cf: float = 80000.0           # front cornering stiffness [N/rad]
    Cr: float = 80000.0           # rear cornering stiffness [N/rad]
    alpha_max: float = 0.15       # soft slip-angle saturation [rad] (~8.6 deg) --
                                   # beyond this the *linear* tire law is no longer
                                   # physically meaningful; a Pacejka ("magic
                                   # formula") tire model is the natural next step.

    @property
    def wheelbase(self) -> float:
        return self.a + self.b


def _tanh_saturate(alpha, alpha_max, backend):
    """Smooth saturation alpha -> alpha_max * tanh(alpha / alpha_max).

    Used (identically, numerically and symbolically) to keep the *linear* tire
    law from producing unbounded lateral force at large slip angles, while
    staying smooth/differentiable -- important so CasADi's NMPC gets clean
    gradients rather than a hard clip's kink.
    """
    return alpha_max * backend.tanh(alpha / alpha_max)


def _smooth_floor(vx, floor, backend, eps=0.05):
    """Smooth (C-infinity) approximation of max(vx, floor):

        0.5 * (vx + floor + sqrt((vx - floor)^2 + eps^2))

    which -> max(vx, floor) as eps -> 0, but has no kink at vx == floor.

    This replaced a hard max()/fmax() that was here originally (guarding the
    atan2() in the slip-angle formula below against a vx ~ 0 singularity). The
    hard version is fine for the numeric "true plant" rollout, but it is a
    real problem for NMPC: whenever the horizon's predicted vx crosses 0.5
    m/s (e.g. accelerating from a standing start), IPOPT has to optimize
    through a non-differentiable kink in its own dynamics constraints, which
    is outside the assumptions its (smooth, gradient/Hessian-based) algorithm
    is built on. In practice this showed up as a dramatic, reproducible
    slowdown -- solves that take ~15-20ms at cruise speed took 500ms-1.5s
    (occasionally hitting the iteration cap outright) while ramping up from a
    stop, because IPOPT needed hundreds of extra iterations to work through
    the kink. Swapping in this smooth floor removes it entirely -- see the
    README for the before/after numbers.
    """
    diff = vx - floor
    return 0.5 * (vx + floor + backend.sqrt(diff * diff + eps * eps))


def dynamics_rhs(state, control, params: VehicleParams, backend=np):
    """Continuous-time right-hand side dx/dt = f(x, u).

    Written against a `backend` module exposing cos/sin/atan2/tanh so the exact
    same equations back both the numeric "true plant" simulation (backend=numpy)
    and the symbolic model CasADi's NMPC optimizes over (backend=casadi) --
    eliminating any chance of the two silently drifting apart.
    """
    X, Y, psi, vx, vy, r = state[0], state[1], state[2], state[3], state[4], state[5]
    ax, delta = control[0], control[1]

    vx_safe = _smooth_floor(vx, 0.5, backend)

    alpha_f = delta - backend.atan2(vy + params.a * r, vx_safe)
    alpha_r = -backend.atan2(vy - params.b * r, vx_safe)
    alpha_f = _tanh_saturate(alpha_f, params.alpha_max, backend)
    alpha_r = _tanh_saturate(alpha_r, params.alpha_max, backend)

    Fyf = params.Cf * alpha_f
    Fyr = params.Cr * alpha_r

    X_dot = vx * backend.cos(psi) - vy * backend.sin(psi)
    Y_dot = vx * backend.sin(psi) + vy * backend.cos(psi)
    psi_dot = r
    vx_dot = ax + vy * r
    vy_dot = (Fyf * backend.cos(delta) + Fyr) / params.mass - vx * r
    r_dot = (params.a * Fyf * backend.cos(delta) - params.b * Fyr) / params.Iz

    if backend is np:
        return np.array([X_dot, Y_dot, psi_dot, vx_dot, vy_dot, r_dot])
    return backend.vertcat(X_dot, Y_dot, psi_dot, vx_dot, vy_dot, r_dot)


class DynamicBicycleModel:
    """Dynamic bicycle model with linear (tanh-saturated) tire forces, RK4
    discretization for simulating the "true" plant."""

    def __init__(self, params: VehicleParams | None = None, dt: float = 0.1,
                 max_speed: float = 20.0, min_speed: float = -3.0,
                 max_accel: float = 3.5, max_steer: float = np.deg2rad(30.0)):
        self.params = params or VehicleParams()
        self.dt = dt
        self.max_speed = max_speed
        self.min_speed = min_speed
        self.max_accel = max_accel
        self.max_steer = max_steer

    @property
    def L(self) -> float:
        return self.params.wheelbase

    def continuous_dynamics(self, state: np.ndarray, control: np.ndarray) -> np.ndarray:
        return dynamics_rhs(state, control, self.params, backend=np)

    def step(self, state: np.ndarray, control: np.ndarray, dt: float | None = None) -> np.ndarray:
        dt = self.dt if dt is None else dt
        control = self._clip_control(control)

        k1 = self.continuous_dynamics(state, control)
        k2 = self.continuous_dynamics(state + dt / 2 * k1, control)
        k3 = self.continuous_dynamics(state + dt / 2 * k2, control)
        k4 = self.continuous_dynamics(state + dt * k3, control)
        next_state = state + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        next_state[3] = np.clip(next_state[3], self.min_speed, self.max_speed)
        return next_state

    def _clip_control(self, control: np.ndarray) -> np.ndarray:
        ax, delta = control
        return np.array([
            np.clip(ax, -self.max_accel, self.max_accel),
            np.clip(delta, -self.max_steer, self.max_steer),
        ])

    def speed(self, state: np.ndarray) -> float:
        """Ground speed magnitude sqrt(vx^2 + vy^2), for reporting/plotting."""
        return float(np.hypot(state[3], state[4]))
