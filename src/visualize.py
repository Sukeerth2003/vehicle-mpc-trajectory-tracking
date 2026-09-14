"""
Visualization: animated GIF of closed-loop tracking (with the MPC predicted
horizon overlaid) and static comparison plots between MPC and the pure
pursuit baseline.
"""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches, transforms
from matplotlib.animation import FuncAnimation, PillowWriter

from simulate import SimResult

CAR_LENGTH = 4.5
CAR_WIDTH = 2.0


def _draw_car(ax, x, y, psi, color, alpha=1.0, zorder=5):
    car = patches.FancyBboxPatch(
        (-CAR_LENGTH / 2, -CAR_WIDTH / 2), CAR_LENGTH, CAR_WIDTH,
        boxstyle="round,pad=0,rounding_size=0.3",
        facecolor=color, edgecolor="black", linewidth=1.0, alpha=alpha, zorder=zorder,
    )
    t = transforms.Affine2D().rotate(psi).translate(x, y) + ax.transData
    car.set_transform(t)
    ax.add_patch(car)
    return car


def animate_tracking(path: np.ndarray, result: SimResult, out_path: str,
                      title: str = "", fps: int = 20, stride: int = 2,
                      show_horizon: bool = True):
    """Render an animated GIF of the vehicle tracking `path` per `result`."""
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Reference path", zorder=1)
    ax.set_aspect("equal")
    margin = 5
    ax.set_xlim(path[:, 0].min() - margin, path[:, 0].max() + margin)
    ax.set_ylim(path[:, 1].min() - margin, path[:, 1].max() + margin)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title or f"{result.controller_name} trajectory tracking")
    ax.legend(loc="upper right", fontsize=8)

    driven_line, = ax.plot([], [], "-", color="#2f6feb", linewidth=2, label="Driven path", zorder=2)
    horizon_line, = ax.plot([], [], "-o", color="#e8590c", markersize=2, linewidth=1.3,
                             alpha=0.9, zorder=3, label="MPC predicted horizon")
    target_pt, = ax.plot([], [], "x", color="#c92a2a", markersize=8, zorder=4, label="Pure pursuit target")

    has_horizon = show_horizon and len(result.predicted_horizons) > 0
    has_target = len(result.targets) > 0
    if not has_horizon:
        horizon_line.set_visible(False)
    if not has_target:
        target_pt.set_visible(False)
    ax.legend(loc="upper right", fontsize=8)

    frames = list(range(0, len(result.states), stride))

    def update(frame_idx):
        i = frames[frame_idx]
        state = result.states[i]
        driven_line.set_data(result.states[:i + 1, 0], result.states[:i + 1, 1])

        for artist in list(ax.patches):
            artist.remove()
        _draw_car(ax, state[0], state[1], state[2], color="#2f6feb")

        if has_horizon and i > 0 and i - 1 < len(result.predicted_horizons):
            h = result.predicted_horizons[i - 1]
            if h is not None:
                horizon_line.set_data(h[0, :], h[1, :])
        if has_target and i > 0 and i - 1 < len(result.targets):
            tp = result.targets[i - 1]
            if tp is not None:
                target_pt.set_data([tp[0]], [tp[1]])

        ax.set_title(f"{title or result.controller_name}  |  t = {result.t[i]:.1f}s  v = {state[3]:.1f} m/s")
        return driven_line, horizon_line, target_pt

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
    plt.close(fig)


def comparison_plot(path: np.ndarray, mpc_result: SimResult, pp_result: SimResult, out_path: str):
    """Static figure: driven paths, lateral error over time, heading error,
    and control inputs, MPC vs pure pursuit."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Reference")
    ax.plot(mpc_result.states[:, 0], mpc_result.states[:, 1], "-", color="#2f6feb", linewidth=2, label="MPC")
    ax.plot(pp_result.states[:, 0], pp_result.states[:, 1], "-", color="#e8590c", linewidth=2, label="Pure Pursuit")
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Driven path vs. reference")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(mpc_result.t[1:], mpc_result.lateral_error, color="#2f6feb", label="MPC")
    ax.plot(pp_result.t[1:], pp_result.lateral_error, color="#e8590c", label="Pure Pursuit")
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("time [s]"); ax.set_ylabel("cross-track error [m]")
    ax.set_title("Lateral (cross-track) tracking error")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(mpc_result.t[1:], np.rad2deg(mpc_result.controls[:, 1]), color="#2f6feb", label="MPC")
    ax.plot(pp_result.t[1:], np.rad2deg(pp_result.controls[:, 1]), color="#e8590c", label="Pure Pursuit")
    ax.set_xlabel("time [s]"); ax.set_ylabel("steering angle [deg]")
    ax.set_title("Control input: steering")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(mpc_result.t[1:], mpc_result.controls[:, 0], color="#2f6feb", label="MPC")
    ax.plot(pp_result.t[1:], pp_result.controls[:, 0], color="#e8590c", label="Pure Pursuit")
    ax.set_xlabel("time [s]"); ax.set_ylabel("acceleration [m/s^2]")
    ax.set_title("Control input: acceleration")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    import os

    from baseline_controller import PurePursuitConfig
    from mpc_controller import MPCConfig
    from simulate import run_mpc, run_pure_pursuit
    from trajectory import get_trajectory
    from vehicle_model import KinematicBicycleModel

    os.makedirs("../results", exist_ok=True)
    model = KinematicBicycleModel(wheelbase=2.7, dt=0.1)
    path = get_trajectory("figure_eight", v_target=5.0, scale=30.0)
    x0 = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0])

    mpc_result = run_mpc(path, model, MPCConfig(), sim_time=40.0, x0=x0.copy())
    pp_result = run_pure_pursuit(path, model, PurePursuitConfig(), sim_time=40.0, x0=x0.copy())

    comparison_plot(path, mpc_result, pp_result, "../results/comparison_plot.png")
    animate_tracking(path, mpc_result, "../results/mpc_tracking.gif",
                      title="LTV-MPC tracking a figure-eight", stride=3)
    animate_tracking(path, pp_result, "../results/pure_pursuit_tracking.gif",
                      title="Pure Pursuit tracking a figure-eight", stride=3)
    print("Saved comparison_plot.png, mpc_tracking.gif, pure_pursuit_tracking.gif to ../results/")
