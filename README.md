# Vehicle Trajectory Tracking with State-Space Modeling and MPC

A from-scratch implementation of **Model Predictive Control (MPC)** for autonomous
vehicle trajectory tracking, in two layers of increasing fidelity:

1. A **kinematic bicycle model** controlled by **Linear Time-Varying MPC** (a convex
   QP, `cvxpy` + `OSQP`), benchmarked against classical **pure pursuit**.
2. A **dynamic bicycle model** (tire slip, lateral forces) controlled by **Nonlinear
   MPC** (`CasADi` + `IPOPT`), extended with **static obstacle avoidance** and stress-
   tested for **robustness to crosswind and sensor/process noise**.

Every result below is measured, not asserted — including the results that don't
flatter this project's own methods (pure pursuit beats MPC on a circle; two real
NMPC convergence bugs found via stress-testing and benchmarking, and how they were
fixed).

![NMPC (dynamic model) tracking a figure-eight](results/nmpc_tracking.gif)

## Why this project

Trajectory tracking sits at the intersection of controls theory, optimization, and
robotics — a bicycle model gives a clean, well-posed state-space system; MPC turns
"drive along this path while respecting actuator limits" into an optimization problem
solved every 100 ms; and a classical baseline (pure pursuit) makes it possible to
show, quantitatively, what the optimization buys you (and where it doesn't). Going
from a kinematic model + convex QP to a dynamic model + nonlinear program is also a
natural, well-trodden path in the controls literature — the project is structured to
show that progression explicitly, including the friction (a real solver-convergence
bug, tuning a hard obstacle constraint that was initially too weak) rather than
presenting only the polished end state.

## Results at a glance

**Part 1 — kinematic model + LTV-MPC vs. pure pursuit:**

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

**Part 2 — dynamic model (tire slip) + NMPC vs. pure pursuit:**

| Trajectory | Controller | Mean \|cross-track error\| | Max \|cross-track error\| |
|---|---|---|---|
| Figure-eight | **NMPC** | **0.035 m** | **0.136 m** |
| Figure-eight | Pure Pursuit | 0.339 m | 1.097 m |
| Double lane change | **NMPC** | **0.011 m** | **0.132 m** |
| Double lane change | Pure Pursuit | 0.052 m | 0.211 m |

Against the more realistic (and harder) dynamic model, NMPC's advantage over pure
pursuit *widens* rather than narrows — pure pursuit is a purely geometric law with no
notion of tire slip, so it has no way to compensate for the vehicle sliding; NMPC
optimizes directly against the true nonlinear dynamics (no linearization at all,
unlike Part 1's LTV-MPC) and corrects for it every step.

**Part 3 — robustness under crosswind + sensor/process noise** (double lane change,
6 Monte Carlo trials per controller, identical disturbance realizations across
controllers):

| Controller | Mean \|cross-track error\| across trials |
|---|---|
| **NMPC** (dynamic model) | **0.113 ± 0.008 m** |
| LTV-MPC (kinematic model) | 0.140 ± 0.007 m |
| Pure Pursuit (dynamic model) | 0.230 ± 0.008 m |
| Pure Pursuit (kinematic model) | 0.236 ± 0.005 m |

Both MPC variants degrade far more gracefully than pure pursuit under disturbance —
their look-ahead lets them start correcting a drift before it compounds, where pure
pursuit only ever reacts to where it's aiming *right now*. NMPC edges out LTV-MPC
here too, consistent with Part 2. See
[Robustness to disturbances](#extension-robustness-to-disturbances) below for the
disturbance model and why the gap is a real, reproducible effect (low trial-to-trial
variance) rather than noise-realization luck.

![Robustness comparison](results/robustness_plot.png)

## Part 1: kinematic model + LTV-MPC

![LTV-MPC tracking a figure-eight](results/mpc_tracking.gif)

### State-space formulation

The vehicle is modeled with the **kinematic bicycle model**, a standard reduced-order
representation of a 4-wheeled vehicle for planning/control at moderate speeds (it
ignores tire slip and lateral dynamics, which matter more at the limits of
handling — see [Part 2](#part-2-dynamic-model--nmpc), which adds exactly this).

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

### The baseline: pure pursuit

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

## Part 2: dynamic model + NMPC

### Why go beyond the kinematic model

The kinematic model assumes **zero tire slip**: the wheels always point exactly
where the vehicle is actually going. That's fine at low speed / low lateral
acceleration, but at higher speeds or tighter turns the tires genuinely slide
sideways relative to their heading, and *that slip is what generates the lateral
force that actually turns the car*. A model that ignores this can't represent, e.g.,
understeer — the same steering angle producing a wider turn radius at higher speed.

### State-space formulation

[`src/dynamic_vehicle_model.py`](src/dynamic_vehicle_model.py) implements the
standard **dynamic bicycle model** with a linear tire law, in the vehicle's
body-fixed frame:

```
x = [X, Y, psi, vx, vy, r]^T      u = [ax, delta]^T
```

(`vx`, `vy` — longitudinal/lateral body-frame speed; `r` — yaw rate.) Front/rear
slip angles are computed from the body-frame velocity components and used with a
linear tire law `Fy = C_alpha * alpha` (smoothly saturated via `tanh` past
`alpha_max` — real tires are linear only for small slip angles):

```
alpha_f = delta - atan2(vy + a*r, vx)          alpha_r = -atan2(vy - b*r, vx)
Fyf = Cf * alpha_f                              Fyr = Cr * alpha_r

X_dot   = vx*cos(psi) - vy*sin(psi)
Y_dot   = vx*sin(psi) + vy*cos(psi)
psi_dot = r
vx_dot  = ax + vy*r
vy_dot  = (Fyf*cos(delta) + Fyr) / m - vx*r
r_dot   = (a*Fyf*cos(delta) - b*Fyr) / Iz
```

where `a`, `b` are the CG-to-front/rear-axle distances and `Cf`, `Cr` the front/rear
cornering stiffnesses. (Reference: Rajamani, *Vehicle Dynamics and Control*, Ch. 2.)

**Validation.** Rather than just trusting the equations, the model is checked against
closed-form theory: for a constant steering angle and speed, a bicycle model has a
known steady-state yaw rate `r_ss = v*delta / (L + Kus*v^2)` (the "understeer
gradient" formula). Simulating the model to steady state (holding forward speed
constant with a small feedforward `ax = -vy*r` that exactly cancels the coupling term
in `vx_dot` — see the note in `validate_dynamic_model.py`, since getting this wrong
is an easy way to silently compare against the wrong operating point) and comparing:
**0.33% error** against the closed-form value — and the resulting turn radius
(33.4 m) is larger than the kinematic model's zero-slip prediction (30.9 m), which is
exactly the understeer effect a kinematic model can't produce. The residual 0.33%
is consistent with the tire model's `tanh` slip-saturation being only *approximately*
linear even at small slip angles (~2-3° here), not numerical error — the closed-form
formula assumes an exactly linear tire law. This, and the NumPy/CasADi bit-identical
dynamics check referenced in the next section, are both in
[`src/validate_dynamic_model.py`](src/validate_dynamic_model.py) (`python
src/validate_dynamic_model.py` reproduces both numbers).

### Nonlinear MPC (NMPC)

The dynamic model's extra states (`vy`, `r`) don't have as clean a "read the
operating point off the reference path" trick as the kinematic model's heading did,
so rather than re-linearize every step (LTV-MPC's approach), the dynamic model is
controlled by a **true nonlinear MPC**: [`src/nmpc_controller.py`](src/nmpc_controller.py)
builds the optimal-control problem symbolically in [CasADi](https://web.casadi.org/)
(direct multiple shooting: both the state trajectory and controls are decision
variables, tied together by RK4-discretized nonlinear dynamics constraints — the
*same* RK4 scheme used to simulate the "true" plant, via a dynamics function shared
between both, verified bit-for-bit identical between its NumPy and CasADi
evaluations) and solves it with IPOPT. The cost structure mirrors Part 1's QP
(tracking + control effort + rate penalty + terminal weight), just without the
linearization.

**A note on NMPC initial guesses (two real bugs found here).** Nonconvex NLPs are
only as reliable as their initial guess, and this project hit two separate, real
failures of that kind while building NMPC — one found by stress-testing under noise,
the other found while double-checking a performance claim for this README.

*Bug 1 — infeasible guesses under sensor noise.* The first version of this
controller seeded IPOPT's state-trajectory guess directly from the reference path's
`[X,Y,psi,v]` values — which don't satisfy the shooting dynamics constraints unless
the vehicle happens to already be exactly on the path. Under sensor noise (Part 3),
that mismatch was enough to blow through the iteration budget on nearly every solve;
worse, the fallback path was reusing IPOPT's last (unconverged, often wildly bad)
iterate as *next step's* warm start, which cascaded the failure for 15-20 consecutive
steps before recovering — a burst of >1 m tracking error that looked like "NMPC is
just worse under noise" until traced back to its actual cause. The fix
(`NMPCController._build_consistent_guess`): forward-simulate the true model from the
*actual* current state using nominal reference-tracking controls, so the initial
guess is dynamically feasible; and on a solve failure, fall back to that consistent
guess rather than the solver's last iterate, so one bad step can't poison the next
one's warm start.

*Bug 2 — a zero-accel guess that could latch forever.* That fix's "nominal
reference-tracking controls" computed a curvature-based steering angle but always
used **zero** acceleration — a reasonable-looking simplification that turned out to
be a real bug. Benchmarking NMPC on the double-lane-change trajectory (which starts
from a dead stop and has to accelerate hard up to 8 m/s) surfaced it: if IPOPT ever
failed to converge while well below the reference speed, it fell back to the
zero-accel guess and returned that guess's (zero-accel) first control *directly* as
`u*` — and because that same guess also became next step's warm start, a single
failed solve could latch the controller into commanding zero acceleration
**permanently**: every following solve was warm-started from, and could fail back
to, the same never-catches-up trajectory. In a from-a-stop test this reliably
stalled the vehicle at a fixed speed forever (`Maximum_Iterations_Exceeded` on every
subsequent solve, control pinned at `[0, 0]`) — the kind of bug that a Monte Carlo
study happens not to surface, since injected noise perturbs the state just enough to
avoid landing in the exact repeating fixed point. The fix: compute the guess's
acceleration *adaptively*, as a simple proportional "close the speed gap" controller
evaluated against the guess trajectory's own evolving speed
(`NMPCController._build_consistent_guess`, replacing the old
`_estimate_nominal_controls`). A related smoothness issue was tightened up at the
same time: the tire model's low-speed guard (`vx_safe = max(vx, 0.5)`, avoiding a
divide-by-zero-adjacent singularity in the slip-angle formula) was a hard `max()`,
which is non-differentiable exactly at `vx = 0.5` — a kink IPOPT's gradient-based
algorithm has to fight through. It's now a smooth approximation
(`dynamic_vehicle_model._smooth_floor`) that agrees with the hard version everywhere
except a small neighborhood of that boundary.

Together, these two fixes are what makes the Part 3 numbers above (and the solve-time
numbers just below) reflect a controller that actually completes every trial rather
than one that got lucky avoiding its own failure mode.

### Obstacle avoidance

The NMPC formulation extends naturally to **non-convex obstacle constraints** —
something a QP-based LTV-MPC can't handle directly without approximation. Each static
circular obstacle `(ox, oy, radius)` adds a hard keep-out constraint over the whole
horizon:

```
(X_k - ox)^2 + (Y_k - oy)^2  >=  radius^2         for every step k of the horizon
```

plus a matching soft penalty that starts "pushing" the cost function away from the
obstacle a bit before the hard boundary — not needed for feasibility (the hard
constraint already guarantees that), but it makes IPOPT converge more reliably by
giving the cost function itself a reason to move away, rather than relying purely on
constraint projection. [`src/obstacle_demo.py`](src/obstacle_demo.py) demonstrates
this on a straight "lane" with two offset static obstacles, comparing the *same* NMPC
controller with the constraint on vs. off:

| Obstacle (X, Y, radius) | Avoidance OFF — closest approach | Avoidance ON — closest approach |
|---|---|---|
| (40.0, 0.4, 1.5 m) | 0.40 m — **collision** | 1.64 m — clear |
| (70.0, -0.6, 1.3 m) | 0.70 m — **collision** | 1.44 m — clear |

![Obstacle avoidance: same NMPC controller, constraint on vs. off](results/obstacle_avoidance_plot.png)

(The first version of this used a *soft-only* penalty, which let the vehicle clip
just inside the safety radius — 1.42 m against a 1.5 m obstacle, technically still a
collision. Switching to a hard constraint, with the soft term kept only as a
convergence aid, fixed it. Another example of a number that looked "close enough"
until actually checked against the stated safety radius.)

### Extension: robustness to disturbances

[`src/robustness_experiment.py`](src/robustness_experiment.py) stress-tests all four
controller/model combinations under three disturbances applied to the *true* plant —
invisible to every controller, which only ever act on a disturbed measurement or
state, never ground truth:

1. **Crosswind** — a constant lateral drift added to world-frame position every step
   (a simplification of a lateral force, documented as such in the code — it's a
   steady disturbance the controller must continuously counter-steer against, not a
   claim of full aerodynamic fidelity).
2. **Process noise** — small Gaussian noise on the true heading and speed every step
   (unmodeled dynamics / actuation noise).
3. **Sensor noise** — Gaussian noise added to the state each controller *measures and
   plans against*; the true plant integrates forward from the real state regardless.

Each controller faces the identical disturbance realization per trial (same RNG
seed), across 6 trials with different seeds, isolating real robustness differences
from noise-realization luck — see the Results table above and
`results/robustness_plot.png`. This experiment is also what surfaced the NMPC
initial-guess bug described above: sensor noise was the specific trigger, since it's
the one disturbance that directly perturbs the state the optimizer's initial guess
has to reconcile with.

## Repository layout

```
.
├── src/
│   ├── vehicle_model.py           # kinematic bicycle model: dynamics, RK4 step, linearization
│   ├── dynamic_vehicle_model.py   # dynamic bicycle model: tire slip, linear tire forces
│   ├── validate_dynamic_model.py  # steady-state cornering + NumPy/CasADi bit-identical checks
│   ├── trajectory.py              # reference path generators + arc-length horizon lookup
│   ├── mpc_controller.py          # LTV-MPC (cvxpy/OSQP) -- kinematic model
│   ├── nmpc_controller.py         # NMPC (CasADi/IPOPT) -- dynamic model, + obstacle avoidance
│   ├── baseline_controller.py     # pure pursuit + PI speed control (works with either model)
│   ├── simulate.py                # closed-loop simulation harness (run_mpc / run_nmpc / run_pure_pursuit)
│   ├── obstacle_demo.py           # obstacle-avoidance scenario + before/after comparison
│   ├── robustness_experiment.py   # crosswind/noise Monte Carlo study across all controllers
│   ├── robustness_plot.py         # bar chart of the robustness study results
│   └── visualize.py               # animated GIFs + comparison plots (shared by all of the above)
├── notebooks/
│   └── demo.ipynb             # walkthrough: derive, simulate, visualize, compare (Part 1)
├── results/                   # generated plots/GIFs (see below to regenerate)
├── app.py                     # interactive Streamlit demo (Part 1: kinematic + LTV-MPC)
└── requirements.txt
```

## Running it

```bash
python -m venv venv && source venv/bin/activate   # optional but recommended
pip install -r requirements.txt

# Part 1: kinematic model + LTV-MPC vs. pure pursuit
python src/simulate.py            # prints tracking error summary
python src/visualize.py           # generates comparison_plot.png + tracking GIFs
streamlit run app.py              # interactive demo (horizon, weights, trajectory, live)

# Part 2: dynamic model + NMPC
python src/validate_dynamic_model.py  # steady-state cornering + NumPy/CasADi dynamics checks
python src/nmpc_demo.py               # NMPC vs. pure pursuit on the figure-eight: comparison_plot + GIF
python src/obstacle_demo.py           # obstacle avoidance: closest-approach numbers + GIF/plot
python src/robustness_experiment.py   # Monte Carlo robustness study (~2-3 min)
python src/robustness_plot.py         # bar chart from the study above
```

**Solve times, measured (not assumed).** "NMPC is slower per-solve than a QP" is the
textbook expectation, but it's worth actually measuring rather than asserting, since
the real picture is more specific than that — and more interesting:

| Trajectory | Controller | Mean | Median | Max |
|---|---|---|---|---|
| Figure-eight | NMPC | 26.8 ms | **24.0 ms** | 737.9 ms |
| Figure-eight | LTV-MPC | 90.2 ms | 81.5 ms | 152.6 ms |
| Double lane change | NMPC | 108.0 ms | **21.2 ms** | 1450.5 ms |
| Double lane change | LTV-MPC | 84.8 ms | 77.9 ms | 131.3 ms |

The *median* NMPC solve is actually faster than the QP's on both trajectories — once
IPOPT is warm-started from a converged previous solution close to the true optimum,
it typically needs only a handful of Newton iterations, cheaper than OSQP re-solving
its QP from scratch every step. But NMPC's *worst case* is far heavier: 5-10x the
QP's max on the figure-eight, over an order of magnitude on the double lane change.
That tail is concentrated in one specific regime — accelerating hard from a standing
start (see the "zero-accel guess" bug above) — where the horizon's dynamics
constraints and the aggressive control demand are genuinely harder for IPOPT to
reconcile, occasionally exhausting the iteration budget outright. The LTV-MPC's QP
solve time barely moves with the scenario, since OSQP's cost per solve doesn't depend
on how hard the maneuver is the way IPOPT's iteration count does.

Practically: if you push this toward real-time deployment, the number that matters is
the *worst case* your control loop's period has to budget for, not the mean — and for
NMPC that worst case is scenario-dependent in a way the QP's isn't. Reproduce these
numbers with `src/simulate.py`'s `SimResult.solve_times` (populated by both
`run_mpc` and `run_nmpc`).

The Streamlit demo (`app.py`) lets you adjust the MPC horizon length, cost weights,
actuator limits, target speed, and trajectory, and re-run the simulation live —
useful for building intuition for the horizon/smoothness/accuracy trade-off (try
shrinking the horizon to 5-6 steps and watch tracking degrade).

## Extensions

This project deliberately started with the simplest model/controller combination
that still produced an honest, working result (Part 1), then grew it deliberately:

- ✅ **Dynamic bicycle model** — tire slip angles and a linear lateral tire force
  model, validated against closed-form steady-state cornering theory (0.33% error).
  See [Part 2](#part-2-dynamic-model--nmpc). A Pacejka ("magic formula") tire model,
  for accuracy beyond the linear region's `alpha_max` (~8.6°), remains a natural
  further step.
- ✅ **Nonlinear MPC (NMPC)** — the true nonlinear dynamics solved directly each
  step via CasADi + IPOPT, instead of re-linearizing. See
  [Part 2](#part-2-dynamic-model--nmpc), including two real convergence bugs found
  and fixed while stress-testing and benchmarking it.
- ✅ **Obstacle avoidance** — hard non-convex keep-out constraints added to the
  NMPC, demonstrated on static obstacles. See
  [Obstacle avoidance](#obstacle-avoidance). Moving obstacles (with a predicted
  trajectory fed into the horizon) are a natural next step.
- ✅ **Robustness** — crosswind + process/sensor noise, Monte Carlo comparison
  across all four controller/model combinations. See
  [Robustness to disturbances](#extension-robustness-to-disturbances). Actuator
  delay and a proper disturbance observer / tube-MPC approach remain open.
- ⬜ **Hardware-in-the-loop.** Port the controller to run in real time against a
  higher-fidelity simulator (e.g. CARLA) or a small RC/robot testbed — the one
  extension from the original plan not yet built here.

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
- J. A. E. Andersson, J. Gillis, G. Horn, J. B. Rawlings, M. Diehl, "CasADi -- A
  software framework for nonlinear optimization and optimal control,"
  *Mathematical Programming Computation*, 2019 — the NMPC implementation here.
- A. Wächter, L. T. Biegler, "On the implementation of an interior-point filter
  line-search algorithm for large-scale nonlinear programming," *Mathematical
  Programming*, 2006 — IPOPT, the NLP solver CasADi calls for NMPC.
