"""
Two validation checks for the dynamic bicycle model + NMPC's internal model,
referenced from the README's "Validation" and "Nonlinear MPC (NMPC)"
sections. Run directly (`python src/validate_dynamic_model.py`) to reproduce
the numbers quoted there.

1. Steady-state cornering vs. closed-form understeer theory: simulate the
   *numeric* model (dynamic_vehicle_model.DynamicBicycleModel) at constant
   speed and steering angle until the yaw rate settles, and compare against
   the standard linear-bicycle steady-state yaw rate formula (Rajamani,
   "Vehicle Dynamics and Control," 2nd ed., eq. 2.30-ish):

       r_ss = v * delta / (L + K_us * v^2),   K_us = (m / L) * (b/Cf - a/Cr)

   This is the same check used to validate the model during development
   (also confirms the model reproduces understeer -- a wider turn radius
   than the kinematic, zero-slip model predicts for the same steering angle).

2. NumPy vs. CasADi bit-identical dynamics: dynamics_rhs() is written once
   against a `backend` argument so the exact same equations back both the
   numeric "true plant" (backend=numpy) and the symbolic model NMPC's IPOPT
   solve optimizes over (backend=casadi). This check evaluates both on a
   batch of random states/controls and confirms they agree to numerical
   precision -- ruling out an entire class of "the model NMPC thinks it's
   controlling doesn't quite match the model actually simulating" bugs
   before they can happen.
"""

import numpy as np

from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams, dynamics_rhs


def validate_steady_state_cornering(v: float = 10.0, delta_deg: float = 5.0,
                                     dt: float = 0.01, n_steps: int = 3000) -> dict:
    """The closed-form formula below assumes a *constant forward speed* v --
    but with ax held at exactly 0, vx does not actually stay at v: the
    vx_dot = ax + vy*r coupling term (vy*r != 0 once the vehicle is
    cornering) makes vx drift upward for as long as the simulation runs, so
    it never reaches the steady state the theory is describing. To hold vx
    at the commanded v (matching the theory's assumption) a small
    feedforward ax = -vy*r is applied every step -- exactly canceling that
    coupling term -- while vy and r themselves are left alone to settle
    wherever the tire forces put them. (This is a real methodological trap,
    not a model bug: an earlier, simpler version of this check omitted the
    compensation, applied ax=0, and let vx drift to ~11.5 m/s well before vy
    and r had actually settled -- comparing the theory's v=10 prediction
    against a simulated state that was quietly no longer at v=10. That gave
    an 11.9% "error" that had nothing to do with the model.)"""
    params = VehicleParams()
    model = DynamicBicycleModel(params=params, dt=dt, max_accel=100.0)
    L = params.wheelbase
    m, Iz, a, b, Cf, Cr = params.mass, params.Iz, params.a, params.b, params.Cf, params.Cr
    delta = np.deg2rad(delta_deg)

    K_us = (m / L) * (b / Cf - a / Cr)
    r_ss_theory = v * delta / (L + K_us * v ** 2)

    state = np.array([0.0, 0.0, 0.0, v, 0.0, 0.0])
    for _ in range(n_steps):
        ax_compensate = -state[4] * state[5]   # cancel vx_dot's vy*r coupling term
        state = model.step(state, np.array([ax_compensate, delta]))
    r_ss_sim = float(state[5])

    err_pct = abs(r_ss_sim - r_ss_theory) / abs(r_ss_theory) * 100

    # Turn radius comparison against the kinematic (zero-slip) model, which
    # has no understeer term: R_kinematic = L / tan(delta).
    R_dynamic = v / abs(r_ss_sim)
    R_kinematic = L / np.tan(delta)

    return {
        "r_ss_theory": r_ss_theory,
        "r_ss_sim": r_ss_sim,
        "error_pct": err_pct,
        "turn_radius_dynamic_m": R_dynamic,
        "turn_radius_kinematic_m": R_kinematic,
    }


def validate_numpy_casadi_match(n_samples: int = 500, seed: int = 0) -> dict:
    import casadi as ca

    params = VehicleParams()
    rng = np.random.default_rng(seed)

    x_sym = ca.MX.sym("x", 6)
    u_sym = ca.MX.sym("u", 2)
    f = ca.Function("f", [x_sym, u_sym], [dynamics_rhs(x_sym, u_sym, params, backend=ca)])

    max_diff = 0.0
    for _ in range(n_samples):
        state = rng.uniform(-5, 15, size=6)
        control = rng.uniform(-2, 2, size=2)
        r_np = dynamics_rhs(state, control, params, backend=np)
        r_ca = np.array(f(state, control)).flatten()
        max_diff = max(max_diff, float(np.max(np.abs(r_np - r_ca))))

    return {"n_samples": n_samples, "max_abs_diff": max_diff}


if __name__ == "__main__":
    print("--- Steady-state cornering vs. closed-form understeer theory ---")
    r1 = validate_steady_state_cornering()
    print(f"v = 10 m/s, delta = 5 deg:")
    print(f"  theory r_ss = {r1['r_ss_theory']:.5f} rad/s")
    print(f"  sim    r_ss = {r1['r_ss_sim']:.5f} rad/s")
    print(f"  error       = {r1['error_pct']:.3f} %")
    print(f"  turn radius: dynamic (with slip) = {r1['turn_radius_dynamic_m']:.2f} m, "
          f"kinematic (zero slip) = {r1['turn_radius_kinematic_m']:.2f} m")

    print("\n--- NumPy vs. CasADi dynamics_rhs() bit-identical check ---")
    r2 = validate_numpy_casadi_match()
    print(f"  {r2['n_samples']} random (state, control) samples: "
          f"max |numpy - casadi| = {r2['max_abs_diff']:.2e}")
