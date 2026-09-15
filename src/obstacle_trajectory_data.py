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


def _simulate(pattern: str, rng: np.random.Generator, n_steps: int, dt: float) -> np.ndarray:
    """Returns (n_steps, 2) array of [x, y] world positions for one instance
    of `pattern`, starting at the origin with a randomized heading/speed/
    maneuver parameters."""
    heading0 = rng.uniform(-np.pi, np.pi)
    speed0 = rng.uniform(1.0, 6.0)   # m/s -- pedestrian to slow-vehicle range

    x, y = 0.0, 0.0
    psi = heading0
    v = speed0
    xs, ys = [x], [y]

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

    else:
        raise ValueError(f"unknown pattern {pattern!r}")

    return np.column_stack([xs, ys])


@dataclass
class ObstacleDataset:
    past_clean: np.ndarray   # (N, K+1, 2) -- observed WORLD positions, noiseless (re-noise on the fly)
    past: np.ndarray         # (N, K, 2) -- per-step displacements, noiseless by default (see add_observation_noise)
    future: np.ndarray       # (N, H, 2) -- cumulative offsets from position at t=0, for h=1..H (always clean)
    patterns: list            # (N,) which pattern each sample came from -- for per-pattern breakdown
    K: int
    H: int
    dt: float


def build_dataset(n_per_pattern: int, K: int = 10, H: int = 10, dt: float = 0.1,
                   seed: int = 0) -> ObstacleDataset:
    """Builds *clean* (noiseless) (past, future) pairs. Sensor noise is added
    separately (see add_observation_noise) rather than baked in here, so
    training can redraw a fresh noise realization every epoch (standard data
    augmentation) instead of overfitting to one frozen noisy sample per
    trajectory -- freezing the noise in the dataset was tried first and
    caused exactly that: near-zero training loss with validation loss
    2-3x worse and *increasing* over training, the classic memorization
    signature. See train_predictor.py for how this is used."""
    rng = np.random.default_rng(seed)
    n_steps = K + H + 1   # +1 so there's a "before the first past step" point to diff against
    past_clean_list, past_list, future_list, pattern_list = [], [], [], []

    for pattern in PATTERNS:
        for _ in range(n_per_pattern):
            traj = _simulate(pattern, rng, n_steps, dt)   # (n_steps, 2)
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

    idx = rng.permutation(len(past_list))
    past_clean = np.array(past_clean_list)[idx]
    past = np.array(past_list)[idx]
    future = np.array(future_list)[idx]
    patterns = [pattern_list[i] for i in idx]

    return ObstacleDataset(past_clean=past_clean, past=past, future=future,
                            patterns=patterns, K=K, H=H, dt=dt)


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
                            patterns=ds.patterns, K=ds.K, H=ds.H, dt=ds.dt)


def split_dataset(ds: ObstacleDataset, train_frac: float = 0.7, val_frac: float = 0.15):
    n = len(ds.past)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    def _slice(a, b):
        return ObstacleDataset(
            past_clean=ds.past_clean[a:b], past=ds.past[a:b], future=ds.future[a:b],
            patterns=ds.patterns[a:b], K=ds.K, H=ds.H, dt=ds.dt,
        )

    return _slice(0, n_train), _slice(n_train, n_train + n_val), _slice(n_train + n_val, n)


if __name__ == "__main__":
    ds = build_dataset(n_per_pattern=50, seed=0)
    print(f"Generated {len(ds.past)} samples ({', '.join(PATTERNS)}), "
          f"K={ds.K} past steps, H={ds.H} future steps, dt={ds.dt}s")
    print("past.shape", ds.past.shape, "future.shape", ds.future.shape)
