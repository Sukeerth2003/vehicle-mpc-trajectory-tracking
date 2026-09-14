"""
Nonlinear MPC (NMPC) for the dynamic bicycle model, via CasADi + IPOPT.

Unlike the LTV-MPC controller (mpc_controller.py), which re-linearizes the
kinematic model about the reference at every step to keep the optimization a
convex QP, NMPC optimizes over the *true* nonlinear dynamics directly ("direct
multiple shooting": the state trajectory and controls are both decision
variables, tied together by RK4-discretized dynamics constraints, matching the
same RK4 scheme used to simulate the "true" plant in dynamic_vehicle_model.py).
This costs more per solve (an NLP, not a QP -- solved here with IPOPT) but
removes the linearization error entirely, which matters more for the dynamic
model since its useful state (vy, r) doesn't have as clean a "read the
operating point off the path" trick as the kinematic model's heading does.

Also supports (optional) static circular obstacle avoidance: a keep-out
constraint ||[X_k,Y_k] - obstacle_center|| >= safety_radius for every step of
the horizon. This is a non-convex constraint that a QP-based LTV-MPC cannot
handle directly (short of convexifying it approximately, e.g. around the
previous solution); IPOPT handles it natively, if only to a local optimum.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import casadi as ca
import numpy as np

from dynamic_vehicle_model import DynamicBicycleModel, N_CONTROLS, N_STATES, dynamics_rhs


@dataclass
class NMPCConfig:
    horizon: int = 10
    Q: np.ndarray = field(default_factory=lambda: np.diag([15.0, 15.0, 3.0, 2.0, 1.0, 1.0]))
    Qf: np.ndarray = field(default_factory=lambda: np.diag([25.0, 25.0, 5.0, 2.0, 1.0, 1.0]))
    R: np.ndarray = field(default_factory=lambda: np.diag([1.0, 15.0]))
    Rd: np.ndarray = field(default_factory=lambda: np.diag([5.0, 30.0]))
    max_steer_rate: float = np.deg2rad(60.0)
    max_accel_rate: float = 4.0
    obstacle_weight: float = 200.0   # soft "start pushing early" penalty weight (see _build_solver())
    ipopt_print_level: int = 0
    ipopt_max_iter: int = 500
    ipopt_acceptable_tol: float = 1e-3   # let IPOPT stop early on a "good enough" iterate
    ipopt_acceptable_iter: int = 5       # ...once it's held for this many consecutive iterations


class NMPCController:
    def __init__(self, model: DynamicBicycleModel, config: NMPCConfig | None = None):
        self.model = model
        self.cfg = config or NMPCConfig()
        self.prev_u = np.zeros(N_CONTROLS)
        self.last_predicted_states: np.ndarray | None = None  # for visualization
        self._prev_solution = None  # warm start: (X_guess, U_guess)
        self._build_solver()

    # ------------------------------------------------------------------
    def _rk4_step(self, x, u, dt):
        f = lambda state, ctrl: dynamics_rhs(state, ctrl, self.model.params, backend=ca)
        k1 = f(x, u)
        k2 = f(x + dt / 2 * k1, u)
        k3 = f(x + dt / 2 * k2, u)
        k4 = f(x + dt * k3, u)
        return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    def _build_solver(self):
        H = self.cfg.horizon
        dt = self.model.dt
        opti = ca.Opti()

        X = opti.variable(N_STATES, H + 1)
        U = opti.variable(N_CONTROLS, H)

        x0_p = opti.parameter(N_STATES)
        xref_p = opti.parameter(N_STATES, H + 1)
        uprev_p = opti.parameter(N_CONTROLS)
        # up to 4 static circular obstacles: [ox, oy, radius] per column; unused
        # slots have radius 0 (see solve()) so their constraint is a no-op.
        obstacles_p = opti.parameter(3, 4)
        obstacle_active_p = opti.parameter(1, 4)

        Q, Qf, R, Rd = self.cfg.Q, self.cfg.Qf, self.cfg.R, self.cfg.Rd

        cost = 0
        opti.subject_to(X[:, 0] == x0_p)
        for k in range(H):
            e = X[:, k] - xref_p[:, k]
            cost += ca.mtimes([e.T, Q, e])
            cost += ca.mtimes([U[:, k].T, R, U[:, k]])

            u_prev_k = uprev_p if k == 0 else U[:, k - 1]
            du = U[:, k] - u_prev_k
            cost += ca.mtimes([du.T, Rd, du])

            opti.subject_to(X[:, k + 1] == self._rk4_step(X[:, k], U[:, k], dt))
            opti.subject_to(opti.bounded(-self.model.max_accel, U[0, k], self.model.max_accel))
            opti.subject_to(opti.bounded(-self.model.max_steer, U[1, k], self.model.max_steer))
            opti.subject_to(opti.bounded(-self.cfg.max_accel_rate * dt, du[0], self.cfg.max_accel_rate * dt))
            opti.subject_to(opti.bounded(-self.cfg.max_steer_rate * dt, du[1], self.cfg.max_steer_rate * dt))

            # Obstacle avoidance: a *hard* keep-out constraint for every active
            # obstacle over the horizon, i.e. ||pos_k - obstacle|| >= radius.
            # Inactive obstacle slots are parked far away (see solve()) so the
            # constraint is trivially satisfied and effectively a no-op for
            # them. We also add a matching soft penalty near the boundary --
            # not for feasibility (the hard constraint already guarantees
            # that) but because IPOPT converges much more reliably when the
            # cost function itself already "wants" to move away from the
            # obstacle, rather than relying solely on constraint projection.
            for j in range(4):
                d2 = (X[0, k] - obstacles_p[0, j]) ** 2 + (X[1, k] - obstacles_p[1, j]) ** 2
                opti.subject_to(d2 >= obstacles_p[2, j] ** 2)
                margin = obstacles_p[2, j] ** 2 * 1.3 - d2   # start "pushing" a bit before the hard boundary
                cost += obstacle_active_p[0, j] * self.cfg.obstacle_weight * ca.fmax(0, margin) ** 2

        e_terminal = X[:, H] - xref_p[:, H]
        cost += ca.mtimes([e_terminal.T, Qf, e_terminal])

        opti.minimize(cost)
        opti.solver("ipopt", {
            "print_time": False,
            "ipopt": {
                "print_level": self.cfg.ipopt_print_level,
                "max_iter": self.cfg.ipopt_max_iter,
                "acceptable_tol": self.cfg.ipopt_acceptable_tol,
                "acceptable_iter": self.cfg.ipopt_acceptable_iter,
                "sb": "yes",
            },
        })

        self._opti = opti
        self._X, self._U = X, U
        self._x0_p, self._xref_p, self._uprev_p = x0_p, xref_p, uprev_p
        self._obstacles_p, self._obstacle_active_p = obstacles_p, obstacle_active_p

    # ------------------------------------------------------------------
    def _estimate_reference_yaw_rate(self, ref_horizon_4col: np.ndarray) -> np.ndarray:
        """r_ref_k ~ d(psi_ref)/dt along the horizon (feedforward yaw rate)."""
        psi = ref_horizon_4col[:, 2]
        dt = self.model.dt
        r = np.diff(psi) / dt
        r = np.append(r, r[-1] if len(r) else 0.0)
        return r

    def _build_full_reference(self, ref_horizon_4col: np.ndarray) -> np.ndarray:
        """[X,Y,psi,v] (from trajectory.reference_horizon) -> full 6-state
        reference [X,Y,psi,vx,vy=0,r] for the dynamic model."""
        r_ref = self._estimate_reference_yaw_rate(ref_horizon_4col)
        vy_ref = np.zeros_like(r_ref)
        return np.column_stack([
            ref_horizon_4col[:, 0], ref_horizon_4col[:, 1], ref_horizon_4col[:, 2],
            ref_horizon_4col[:, 3], vy_ref, r_ref,
        ])

    def _estimate_nominal_steering(self, ref_horizon_4col: np.ndarray) -> np.ndarray:
        """Curvature-based nominal steering (same trick as the LTV-MPC's
        _estimate_reference_controls) -- "roughly what a reasonable driver
        would do to steer along this path", used by _build_consistent_guess.
        Doesn't need to be exact for the dynamic model's slip -- it only has
        to be good enough that IPOPT starts from a near-feasible trajectory
        rather than an arbitrary one. (The matching accel guess is computed
        separately, adaptively, in _build_consistent_guess -- see the note
        there on why a fixed zero-accel guess is not good enough.)"""
        psi = ref_horizon_4col[:, 2]
        v = ref_horizon_4col[:, 3]
        dpsi = np.diff(psi)
        ds = np.maximum(v[:-1] * self.model.dt, 1e-3)
        kappa = dpsi / ds
        delta_ref = np.arctan(self.model.L * kappa)
        delta_ref = np.clip(delta_ref, -self.model.max_steer, self.model.max_steer)
        delta_ref = np.append(delta_ref, delta_ref[-1] if len(delta_ref) else 0.0)
        return delta_ref

    def _build_consistent_guess(self, x0: np.ndarray, ref_horizon_4col: np.ndarray):
        """A *dynamically consistent* initial guess for (X, U): forward-
        simulate the true (numpy) model from the actual x0 using nominal
        reference-tracking controls, via the exact same RK4 integration
        scheme used both by the true plant and by the NLP's shooting
        constraints. This matters a lot for convergence reliability --
        seeding IPOPT with X values taken straight from the reference path
        (which generally do NOT satisfy x_{k+1} = f(x_k, u_k) starting from
        the *actual* current x0, e.g. whenever x0 is noisy or simply not on
        the path) forces the solver to reconcile a badly infeasible starting
        point, which can blow through the iteration budget rather than
        converge -- exactly the failure mode that motivated this function
        (see the "note on NMPC initial guesses" in the README).

        The acceleration part of the guess is computed *adaptively* here
        (rather than being fixed at zero, as an earlier version of this
        function did): a_ref_k = clip((v_ref_k - v_k) / dt, +/-max_accel), a
        simple proportional "close the speed gap" controller evaluated
        against the guess trajectory's own evolving speed, not just x0's.
        A constant zero-accel guess turned out to be a real bug, not just an
        approximation: whenever the current speed was well below the
        reference (e.g. accelerating from a stop) *and* IPOPT failed to
        converge, the solver fell back to this guess and returned its
        (zero-accel) first control directly as u* -- and because that same
        guess also becomes next step's warm start, a single failed solve
        could latch the controller into commanding zero acceleration
        forever, since every subsequent solve was warm-started from, and
        could fail back to, the same never-catches-up trajectory. Making the
        guess actually try to close the speed gap breaks that latch. See the
        README for the before/after."""
        H = self.cfg.horizon
        delta_ref = self._estimate_nominal_steering(ref_horizon_4col)
        v_ref = ref_horizon_4col[:, 3]
        dt = self.model.dt
        max_accel = self.model.max_accel

        X_guess = np.zeros((N_STATES, H + 1))
        U_guess = np.zeros((N_CONTROLS, H))
        X_guess[:, 0] = x0
        x = x0.copy()
        for k in range(H):
            a_k = np.clip((v_ref[k] - x[3]) / dt, -max_accel, max_accel)
            u_k = np.array([a_k, delta_ref[k]])
            U_guess[:, k] = u_k
            x = self.model.step(x, u_k)
            X_guess[:, k + 1] = x
        return X_guess, U_guess

    def solve(self, x0: np.ndarray, ref_horizon_4col: np.ndarray,
              obstacles: list[tuple[float, float, float]] | None = None) -> np.ndarray:
        """ref_horizon_4col: (H+1, 4) array of [X,Y,psi,v] (same format
        trajectory.reference_horizon returns for the kinematic/LTV-MPC
        controller). obstacles: optional list of (ox, oy, radius) static
        circular keep-out zones -- see _build_solver for how these enter as
        hard constraints. Returns u* = [ax, delta]."""
        H = self.cfg.horizon
        ref6 = self._build_full_reference(ref_horizon_4col)
        # NOTE: the dynamically-consistent guess (_build_consistent_guess) is a
        # full numpy forward-simulation over the horizon -- real work, not
        # free. It's only actually needed (a) the very first call, before any
        # warm start exists, or (b) as a safe fallback if IPOPT fails to
        # converge. Computing it unconditionally on every call (as an earlier
        # version of this code did) added that cost to every single step even
        # when a perfectly good warm start was already available, which is
        # why an early benchmark showed NMPC solves several times slower than
        # they needed to be -- so it's computed lazily below instead.

        # Inactive obstacle slots: radius 0 makes the hard constraint
        # d^2 >= radius^2 trivially true everywhere (position is irrelevant),
        # which is more robust than "parking far away" with a nonzero radius
        # would be (that leaves d^2 needing to also be astronomically large).
        obs_arr = np.zeros((3, 4))
        obs_active = np.zeros((1, 4))
        if obstacles:
            for j, (ox, oy, rad) in enumerate(obstacles[:4]):
                obs_arr[:, j] = [ox, oy, rad]
                obs_active[0, j] = 1.0

        opti = self._opti
        opti.set_value(self._x0_p, x0)
        opti.set_value(self._xref_p, ref6.T)
        opti.set_value(self._uprev_p, self.prev_u)
        opti.set_value(self._obstacles_p, obs_arr)
        opti.set_value(self._obstacle_active_p, obs_active)

        if self._prev_solution is not None:
            X_guess, U_guess = self._prev_solution
            opti.set_initial(self._X, X_guess)
            opti.set_initial(self._U, U_guess)
        else:
            X_guess, U_guess = self._build_consistent_guess(x0, ref_horizon_4col)
            opti.set_initial(self._X, X_guess)
            opti.set_initial(self._U, U_guess)

        try:
            sol = opti.solve()
            X_opt = sol.value(self._X)
            U_opt = sol.value(self._U)
        except RuntimeError:
            # IPOPT failed to converge (e.g. hit max_iter) -- deliberately do
            # NOT fall back to opti.debug.value() here. That "last iterate"
            # can be an arbitrarily bad, dynamically-inconsistent point this
            # deep into a failed nonconvex solve, and -- worse -- feeding it
            # back in as next step's warm start tends to cascade the failure
            # for many steps in a row (this was a real bug found while
            # stress-testing under sensor noise; see the README). Falling
            # back to the dynamically-consistent forward-simulated guess is
            # a safe, bounded-quality substitute: not optimal, but a valid
            # trajectory that won't poison future warm starts. Only computed
            # here, on the (rare) failure path -- see the note above solve().
            X_opt, U_opt = self._build_consistent_guess(x0, ref_horizon_4col)

        # warm start next call: shift the horizon by one step
        X_shifted = np.hstack([X_opt[:, 1:], X_opt[:, -1:]])
        U_shifted = np.hstack([U_opt[:, 1:], U_opt[:, -1:]])
        self._prev_solution = (X_shifted, U_shifted)

        self.last_predicted_states = X_opt
        u_opt = U_opt[:, 0]
        self.prev_u = u_opt.copy()
        return u_opt
