"""
Synthetic moving-obstacle trajectory dataset.

There's no real traffic/pedestrian dataset wired into this project, so the
predictor in ssm_predictor.py is trained on synthetic motion patterns
generated here instead. Four qualitatively different patterns are used, so
the model has to learn *general* short-horizon motion prediction rather than
overfitting a single maneuver shape:

  1. constant_velocity  -- straight line, fixed speed (e.g. a car cruising
                            across the ego vehicle's path).
  2. decelerating        -- straight line, slowing to a stop partway through
                            (e.g. a pedestrian stopping at a curb, or a
                            vehicle braking).
  3. constant_turn        -- constant turn-rate arc (e.g. a vehicle turning).
  4. weave                -- forward motion with a sinusoidal lateral
                            oscillation on top (e.g. an erratic
                            cyclist/pedestrian) -- deliberately the hardest
                            pattern for a constant-velocity assumption to
                            track, and the one where a learned model should
                            show the clearest advantage.

A fifth pattern, "branch", is available (via the `patterns=` argument to
build_dataset) but deliberately excluded from the default PATTERNS list used
by the Part 3 unimodal predictor, since it is not fair to that model: for
every one of the K observed steps, "branch" is statistically IDENTICAL to
constant_velocity -- the obstacle either keeps going at constant velocity or
decelerates hard to a stop, decided by a coin flip that is drawn independent
of, and takes effect strictly after, the observed window. No information in
the past K steps can possibly reveal which outcome is coming; a unimodal
(single-point) predictor is mathematically stuck predicting something between
the two true futures, which matches neither. This is exactly the case a
*mixture* predictor (see ssm_predictor.MultimodalObstaclePredictor and
train_multimodal_predictor.py) is for, and exactly why it's kept out of Part
3's training data rather than silently changing those already-reported
results.

Every trajectory is generated in the obstacle's own world frame at dt = 0.1s
(matching the rest of this project). The dataset is built as (past, future)
pairs for sequence-to-sequence forecasting: given K observed steps, predict
the next H steps -- both expressed as *offsets* from the last observed
position (not absolute world coordinates), so the model learns
location-invariant motion patterns rather than memorizing where obstacles
happened to be during training. See `build_dataset`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

PATTERNS = ["constant_velocity", "decelerating", "constant_turn", "weave"]
AMBIGUOUS_PATTERN = "branch"
ALL_PATTERNS = PATTERNS + [AMBIGUOUS_PATTERN]


def _simulate(pattern: str, rng: np.random.Generator, n_steps: int, dt: float,
              K: int | None = None):
    """Returns (traj, meta): traj is (n_steps, 2) [x, y] world positions for
    one instance of `pattern`, starting at the origin with a randomized
    heading/speed/maneuver parameters; meta is a dict of extra per-sample
    info (currently only {"branch": "go"|"stop"} for the "branch" pattern,
    {} otherwise). `K` is only used by "branch", to know exactly where the
    observed window ends and the branch point begins."""
    heading0 = rng.uniform(-np.pi, np.pi)
    speed0 = rng.uniform(1.0, 6.0)   # m/s -- pedestrian to slow-vehicle range

    x, y = 0.0, 0.0
    psi = heading0
    v = speed0
    xs, ys = [x], [y]
    meta = {}

    if pattern == "constant_velocity":
        for _ in range(n_steps - 1):
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            xs.append(x); ys.append(y)

    elif pattern == "decelerating":
        decel_start = rng.integers(n_steps // 4, n_steps // 2)
        decel = rng.uniform(0.5, 2.5)   # m/s^2
        for k in range(n_steps - 1):
            if k >= decel_start:
                v = max(0.0, v - decel * dt)
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            xs.append(x); ys.append(y)

    elif pattern == "constant_turn":
        turn_rate = rng.uniform(0.2, 0.8) * rng.choice([-1.0, 1.0])   # rad/s
        for _ in range(n_steps - 1):
            psi += turn_rate * dt
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            xs.append(x); ys.append(y)

    elif pattern == "weave":
        amp = rng.uniform(0.3, 1.2)      # m, lateral weave amplitude
        freq = rng.uniform(0.3, 1.0)     # Hz
        phase = rng.uniform(0, 2 * np.pi)
        t = 0.0
        for _ in range(n_steps - 1):
            t += dt
            lateral_rate = amp * 2 * np.pi * freq * np.cos(2 * np.pi * freq * t + phase)
            local_psi = psi + np.arctan2(lateral_rate, v)
            x += v * np.cos(local_psi) * dt
            y += v * np.sin(local_psi) * dt
            xs.append(x); ys.append(y)

    elif pattern == "branch":
        if K is None:
            raise ValueError("branch pattern requires K (the observed-window length)")
        # Genuinely ambiguous: constant velocity for every one of the K
        # observed steps (indices 0..K), branching only from step K onward
        # (the first *future* step) -- a coin flip drawn here, with no
        # influence on anything before it, so the observed window carries
        # zero information about which way it will go.
        #
        # speed0/decel are deliberately narrower and faster here than the
        # generic pedestrian-to-slow-vehicle range used above: the first
        # version of this pattern reused the full speed0 in [1, 6] m/s
        # range, and for the faster samples the H=10-step (1s) horizon
        # wasn't long enough for "stop" to diverge visibly from "go" --
        # decelerating from 6 m/s takes ~2s even at max decel, so within
        # the horizon it looked almost identical to constant velocity,
        # muddying the training signal (the multimodal predictor's mode
        # usage stayed close to 50/50 *within* both true branches instead
        # of specializing). Keeping speed0 slow enough and decel sharp
        # enough that "stop" reliably resolves within the horizon is what
        # makes this pattern actually teach bimodality rather than just
        # add noise. See train_multimodal_predictor.py's branch
        # diagnostics for how this is checked, not just assumed.
        speed0 = rng.uniform(1.0, 3.0)
        v = speed0
        branch = "stop" if rng.random() < 0.5 else "go"
        decel = rng.uniform(3.0, 5.0)
        for k in range(n_steps - 1):
            if branch == "stop" and k >= K:
                v = max(0.0, v - decel * dt)
            x += v * np.cos(psi) * dt
            y += v * np.sin(psi) * dt
            xs.append(x); ys.append(y)
        meta = {"branch": branch}

    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    return np.column_stack([xs, ys]), meta


@dataclass
class ObstacleDataset:
    past_clean: np.ndarray   # (N, K+1, 2) -- observed WORLD positions, noiseless (re-noise on the fly)
    past: np.ndarray         # (N, K, 2) -- per-step displacements, noiseless by default (see add_observation_noise)
    future: np.ndarray       # (N, H, 2) -- cumulative offsets from position at t=0, for h=1..H (always clean)
    patterns: list            # (N,) which pattern each sample came from -- for per-pattern breakdown
    K: int
    H: int
    dt: float
    branches: list = field(default_factory=list)   # (N,) "go"/"stop" for the "branch" pattern, "" otherwise


def build_dataset(n_per_pattern: int, K: int = 10, H: int = 10, dt: float = 0.1,
                   seed: int = 0, patterns: list | None = None) -> ObstacleDataset:
    """Builds *clean* (noiseless) (past, future) pairs. Sensor noise is added
    separately (see add_observation_noise) rather than baked in here, so
    training can redraw a fresh noise realization every epoch (standard data
    augmentation) instead of overfitting to one frozen noisy sample per
    trajectory -- freezing the noise in the dataset was tried first and
    caused exactly that: near-zero training loss with validation loss
    2-3x worse and *increasing* over training, the classic memorization
    signature. See train_predictor.py for how this is used.

    `patterns` defaults to PATTERNS (the four unimodal patterns, unchanged
    behavior for Part 3's predictor). Pass `patterns=ALL_PATTERNS` (or any
    list including "branch") to also include the genuinely ambiguous
    stop-or-go pattern, e.g. for train_multimodal_predictor.py -- kept
    opt-in rather than default so it never silently changes the unimodal
    predictor's already-reported training data/results."""
    patterns = patterns or PATTERNS
    rng = np.random.default_rng(seed)
    n_steps = K + H + 1   # +1 so there's a "before the first past step" point to diff against
    past_clean_list, past_list, future_list, pattern_list, branch_list = [], [], [], [], []

    for pattern in patterns:
        for _ in range(n_per_pattern):
            traj, meta = _simulate(pattern, rng, n_steps, dt, K=K)   # (n_steps, 2)
            # index convention: traj[0 .. K] are the K+1 points needed to get
            # K per-step displacements ending "now" (t=0, index K); traj[K:]
            # are the H future positions.
            observed = traj[: K + 1].copy()
            now = traj[K]
            past_disp = np.diff(observed, axis=0)                 # (K, 2)
            future_offset = traj[K + 1: K + 1 + H] - now          # (H, 2)
            past_clean_list.append(observed)
            past_list.append(past_disp)
            future_list.append(future_offset)
            pattern_list.append(pattern)
            branch_list.append(meta.get("branch", ""))

    idx = rng.permutation(len(past_list))
    past_clean = np.array(past_clean_list)[idx]
    past = np.array(past_list)[idx]
    future = np.array(future_list)[idx]
    patterns_out = [pattern_list[i] for i in idx]
    branches_out = [branch_list[i] for i in idx]

    return ObstacleDataset(past_clean=past_clean, past=past, future=future,
                            patterns=patterns_out, K=K, H=H, dt=dt, branches=branches_out)


def add_observation_noise(ds: ObstacleDataset, std: float, rng: np.random.Generator) -> ObstacleDataset:
    """Returns a copy of `ds` with fresh Gaussian position noise (meters,
    std `std`) applied to the observed window, re-derived into per-step
    displacements -- everything else (future ground truth) unchanged. This
    mirrors real perception noise: you never observe another agent's true
    position, only a noisy estimate of their recent track, and must predict
    their true future path from that. Call with a fresh `rng` draw each
    training epoch for augmentation, or once (seeded) for a reproducible
    evaluation set."""
    if std <= 0:
        return ds
    noisy_observed = ds.past_clean + rng.normal(0, std, size=ds.past_clean.shape)
    noisy_past = np.diff(noisy_observed, axis=1)
    return ObstacleDataset(past_clean=ds.past_clean, past=noisy_past, future=ds.future,
                            patterns=ds.patterns, K=ds.K, H=ds.H, dt=ds.dt, branches=ds.branches)


def split_dataset(ds: ObstacleDataset, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(ds.past)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    def _slice(a, b):
        return ObstacleDataset(
            past_clean=ds.past_clean[a:b], past=ds.past[a:b], future=ds.future[a:b],
            patterns=ds.patterns[a:b], K=ds.K, H=ds.H, dt=ds.dt, branches=ds.branches[a:b],
        )

    return _slice(0, n_train), _slice(n_train, n_train + n_val), _slice(n_train + n_val, n)


if __name__ == "__main__":
    ds = build_dataset(n_per_pattern=50, seed=0)
    print(f"Generated {len(ds.past)} samples ({', '.join(PATTERNS)}), "
          f"K={ds.K} past steps, H={ds.H} future steps, dt={ds.dt}s")
    print("past.shape", ds.past.shape, "future.shape", ds.future.shape)

    ds_amb = build_dataset(n_per_pattern=20, seed=1, patterns=ALL_PATTERNS)
    branch_mask = np.array([p == "branch" for p in ds_amb.patterns])
    go_mask = branch_mask & (np.array(ds_amb.branches) == "go")
    stop_mask = branch_mask & (np.array(ds_amb.branches) == "stop")
    # Sanity check: past displacements for "go" and "stop" branch samples
    # should be drawn from the same distribution (they're both just
    # constant-velocity segments) -- confirm the observed windows don't
    # trivially leak the branch, e.g. via a mean-past-speed check.
    go_speed = np.linalg.norm(ds_amb.past[go_mask], axis=-1).mean()
    stop_speed = np.linalg.norm(ds_amb.past[stop_mask], axis=-1).mean()
    print(f"\nbranch pattern: {branch_mask.sum()} samples "
          f"({go_mask.sum()} go, {stop_mask.sum()} stop)")
    print(f"mean observed per-step speed -- go: {go_speed:.3f} m  stop: {stop_speed:.3f} m "
          "(should be close: the observed window is branch-independent)")
