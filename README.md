# Vehicle Trajectory Tracking with State-Space Modeling and MPC

A from-scratch implementation of **Model Predictive Control (MPC)** for autonomous
vehicle trajectory tracking, built on a **state-space kinematic bicycle model**. The
project derives the model and MPC formulation from first principles, implements a
Linear Time-Varying MPC (LTV-MPC) controller as a convex QP (`cvxpy` + `OSQP`),
and benchmarks it against a classical **pure pursuit** controller across three
reference maneuvers.

![MPC tracking a figure-eight](results/mpc_tracking.gif)

## Why this project

Trajectory tracking sits at the intersection of controls theory, optimization, and
robotics — a kinematic bicycle model gives a clean, well-posed state-space system;
MPC turns "drive along this path while respecting actuator limits" into a QP solved
every 100 ms; and a classical baseline (pure pursuit) makes it possible to show,
quantitatively, what the optimization buys you (and where it doesn't).

## Results at a glance

| Trajectory | Controller | Mean \|cross-track error\| | Max \|cross-track error\| |
|---|---|---|---|
| Figure-eight (tight, varying curvature) | **MPC** | **0.087 m** | **0.261 m** |
| Figure-eight | Pure Pursuit | 0.164 m | 0.486 m |
| Double lane change (ISO 3888-style) | **MPC** | **0.013 m** | **0.067 m** |
| Double lane change | Pure Pursuit | 0.059 m | 0.227 m |
| Constant-radius circle | MPC | 0.120 m | 0.248 m |
| Constant-radius circle | **Pure Pursuit** | **0.009 m** | **0.011 m** |

The interesting result isn't "MPC always wins" — it's *where* it wins. MPC's
look-ahead over the horizon lets it anticipate curvature and speed changes, so it
clearly outperforms pure pursuit on the figure-eight and the lane change, both of
which demand reacting to upcoming geometry. On a constant-radius circle, though,
pure pursuit's simple geometric law is a near-perfect match for the path shape, and
edges MPC out. That trade-off — a model-based optimizer's generality vs. a
classical controller's fit to a specific geometry — is a big part of why MPC is used
in practice for the harder cases.

## The model

### State-space formulation

The vehicle is modeled with the **kinematic bicycle model**, a standard reduced-order
representation of a 4-wheeled vehicle for planning/control at moderate speeds (it
ignores tire slip and lateral dynamics, which matter more at the limits of
handling — see [Extensions](#extensions-ideas-for-going-further)).

State vector (position and heading of the rear axle, plus speed):

```
x = [X, Y, psi, v]^T
```

Control input (longitudinal acceleration and front steering angle):

```
u = [a, delta]^T
```

Continuous-time nonlinear dynamics:

```
X_dot   = v * cos(psi)
Y_dot   = v * sin(psi)
psi_dot = (v / L) * tan(delta)
v_dot   = a
```

where `L` is the wheelbase. This is implemented in
[`src/vehicle_model.py`](src/vehicle_model.py), integrated with RK4 for the "true"
plant simulation.

### Linearization for MPC (LTV-MPC)

The dynamics above are nonlinear (through `cos(psi)`, `sin(psi)`, `tan(delta)`), so
directly embedding them in a QP isn't convex. The standard trick — and what this
project does — is to **linearize about a reference (operating) point at every
control step**, producing a time-varying linear system that's re-derived each time
the MPC solves:

```
A_c = d(f)/d(x) |_(x_bar, u_bar),      B_c = d(f)/d(u) |_(x_bar, u_bar)
```

Discretized with a first-order (Euler) hold at sample time `dt`:

```
A_d = I + dt * A_c
B_d = dt * B_c
C_d = dt * (f(x_bar, u_bar) - A_c @ x_bar - B_c @ u_bar)     # affine correction term
```

so that each step of the horizon obeys `x_{k+1} = A_d x_k + B_d u_k + C_d`. Because
the true dynamics are nonlinear, this local linearization is only exact at the
operating point — the affine term `C_d` corrects for the mismatch to first order, and
re-linearizing every control step (rather than once, offline) is what keeps the
approximation valid as the vehicle moves. The operating point at each step of the
horizon is taken from the reference path itself (position, heading, and a nominal
steering angle estimated from the path's local curvature, `delta_ref = atan(L * kappa)`),
so the linearization tracks the maneuver being commanded, not just the vehicle's
current pose. See [`KinematicBicycleModel.linearize`](src/vehicle_model.py) and
[`MPCController._estimate_reference_controls`](src/mpc_controller.py).

**A note on heading wraparound.** `psi` is kept as an unbounded, continuous real
number throughout the codebase rather than wrapped into `[-pi, pi]`. The dynamics
don't care (`sin`/`cos` are periodic), but the MPC's quadratic cost on heading error
does: if `psi` were wrapped, a reference heading of `-3.10 rad` and a vehicle heading
of `+3.10 rad` — physically 0.08 rad apart — would appear ~6.2 rad apart to the cost
function, and the controller would fight to "correct" an error that doesn't exist.
Every reference trajectory in [`src/trajectory.py`](src/trajectory.py) uses
`np.unwrap` for exactly this reason. (This is a real bug I hit and fixed while
building this project — worth knowing about if you extend it.)

### MPC formulation

At every control step, given the current state `x0` and a horizon of `H` reference
points, the controller solves:

```
minimize    sum_{k=0}^{H-1}  (x_k - x_ref_k)' Q (x_k - x_ref_k)
                            + u_k' R u_k
                            + (u_k - u_{k-1})' Rd (u_k - u_{k-1})
          + (x_H - x_ref_H)' Qf (x_H - x_ref_H)

subject to  x_{k+1} = A_k x_k + B_k u_k + C_k        for k = 0..H-1
            x_0 = current state
            |a_k|      <= a_max
            |delta_k|  <= delta_max
            |a_k - a_{k-1}|         <= da_max * dt
            |delta_k - delta_{k-1}| <= ddelta_max * dt
```

`Q` penalizes state tracking error, `R` penalizes control effort, `Rd` penalizes
control *rate* (this is what keeps the steering from chattering), and `Qf` is a
heavier terminal weight to pull the end of the horizon toward the path. Only the
first control input `u_0*` is applied (receding-horizon control); the whole QP is
re-solved next step with the updated true state. See
[`MPCController.solve`](src/mpc_controller.py).

The reference horizon itself is built by arc-length lookup (not by raw path-sample
index) — `H` points spaced by how far the vehicle would actually travel in `dt` at
the reference speed — so the horizon's real-world length in meters doesn't silently
depend on how densely a trajectory happens to be sampled. See
[`trajectory.reference_horizon`](src/trajectory.py).

## The baseline: pure pursuit

[`src/baseline_controller.py`](src/baseline_controller.py) implements
[pure pursuit](https://www.ri.cmu.edu/pub_files/pub3/coulter_r_craig_1992_1/coulter_r_craig_1992_1.pdf):
pick a lookahead point on the path a fixed distance ahead, and steer to intersect it
along a circular arc through the vehicle's current pose —
`delta = atan2(2 * L * sin(alpha), Ld)`, where `alpha` is the angle to the lookahead
point and `Ld` the lookahead distance. A simple PI loop handles longitudinal speed
tracking. It has no notion of a prediction horizon or actuator-rate limits; it reacts
to *where it's aiming right now*, which is exactly why it does well on a circle
(constant curvature = the geometry pure pursuit is implicitly built around) and worse
on maneuvers where the "right" curvature keeps changing (the figure-eight, the lane
change).

## Repository layout

```
.
├── src/
│   ├── vehicle_model.py       # kinematic bicycle model: dynamics, RK4 step, linearization
│   ├── trajectory.py          # reference path generators + arc-length horizon lookup
│   ├── mpc_controller.py      # LTV-MPC (cvxpy/OSQP)
│   ├── baseline_controller.py # pure pursuit + PI speed control
│   ├── simulate.py            # closed-loop simulation harness
│   └── visualize.py           # animated GIF + comparison plots
├── notebooks/
│   └── demo.ipynb             # walkthrough: derive, simulate, visualize, compare
├── results/                   # generated plots/GIFs (see below to regenerate)
├── app.py                     # interactive Streamlit demo
└── requirements.txt
```

## Running it

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Run a simulation + print tracking error summary
python src/simulate.py

# Generate the comparison plot + tracking GIFs into results/
python src/visualize.py

# Launch the interactive demo
streamlit run app.py
```

The Streamlit demo (`app.py`) lets you adjust the MPC horizon length, cost weights,
actuator limits, target speed, and trajectory, and re-run the simulation live —
useful for building intuition for the horizon/smoothness/accuracy trade-off (try
shrinking the horizon to 5-6 steps and watch tracking degrade).

## Extensions (ideas for going further)

This project deliberately starts with the simplest model/controller combination that
still produces an honest, working result, with room to grow:

- **Dynamic bicycle model.** Add tire slip angles and a lateral tire force model
  (e.g. linear or Pacejka "magic formula" tires) for accuracy at higher speeds and
  more aggressive maneuvers, where the kinematic assumption (zero slip) breaks down.
- **Nonlinear MPC (NMPC).** Solve the true nonlinear dynamics directly each step
  (e.g. with [CasADi](https://web.casadi.org/) + IPOPT, or
  [do-mpc](https://www.do-mpc.com/)) instead of re-linearizing — more accurate,
  more expensive, and a natural next step once LTV-MPC's approximation starts to
  show its limits.
- **Obstacle avoidance.** Add convex (or convexified) obstacle constraints to the
  QP, or a soft-constraint penalty, and test on a path with static/moving obstacles.
- **Robustness.** Add process/measurement noise and disturbances (e.g. a
  crosswind term, actuator delay) and compare MPC's and pure pursuit's degradation
  under uncertainty — a natural lead-in to robust or stochastic MPC.
- **Hardware-in-the-loop.** Port the controller to run in real time against a
  higher-fidelity simulator (e.g. CARLA) or a small RC/robot testbed.

## References

- R. Rajamani, *Vehicle Dynamics and Control*, Springer, 2011 — kinematic/dynamic
  bicycle models.
- J. Kong et al., "Kinematic and dynamic vehicle models for autonomous driving
  control design," *IEEE Intelligent Vehicles Symposium*, 2015 — the LTV-MPC
  approach used here.
- R. C. Coulter, "Implementation of the Pure Pursuit Path Tracking Algorithm,"
  CMU Robotics Institute Technical Report, 1992.
- J. B. Rawlings, D. Q. Mayne, M. Diehl, *Model Predictive Control: Theory,
  Computation, and Design*, 2nd ed., Nob Hill Publishing, 2017.
