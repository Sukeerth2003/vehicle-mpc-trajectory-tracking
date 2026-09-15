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
                      show_horizon: bool = True,
                      obstacles: list[tuple[float, float, float]] | None = None,
                      xlim: tuple[float, float] | None = None,
                      ylim: tuple[float, float] | None = None,
                      equal_aspect: bool = True,
                      figsize: tuple[float, float] = (7, 7)):
    """Render an animated GIF of the vehicle tracking `path` per `result`.
    `obstacles`, if given, are drawn as filled circles: [(X, Y, radius), ...].
    Set `equal_aspect=False` for scenarios (like a long straight lane) where a
    true 1:1 aspect ratio would squash the plot into an unreadable sliver."""
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Reference path", zorder=1)
    if equal_aspect:
        ax.set_aspect("equal")
    margin = 5
    ax.set_xlim(*(xlim or (path[:, 0].min() - margin, path[:, 0].max() + margin)))
    ax.set_ylim(*(ylim or (path[:, 1].min() - margin, path[:, 1].max() + margin)))
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(title or f"{result.controller_name} trajectory tracking")

    for i, (ox, oy, orad) in enumerate(obstacles or []):
        ax.add_patch(patches.Circle((ox, oy), orad, facecolor="#f03e3e", edgecolor="#9a1414",
                                     alpha=0.5, zorder=2, label="Obstacle" if i == 0 else None))

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
    car_patch_holder = {"patch": None}

    def update(frame_idx):
        i = frames[frame_idx]
        state = result.states[i]
        driven_line.set_data(result.states[:i + 1, 0], result.states[:i + 1, 1])

        if car_patch_holder["patch"] is not None:
            car_patch_holder["patch"].remove()
        car_patch_holder["patch"] = _draw_car(ax, state[0], state[1], state[2], color="#2f6feb")

        if has_horizon and i > 0 and i - 1 < len(result.predicted_horizons):
            h = result.predicted_horizons[i - 1]
            if h is not None:
                horizon_line.set_data(h[0, :], h[1, :])
        if has_target and i > 0 and i - 1 < len(result.targets):
            tp = result.targets[i - 1]
            if tp is not None:
                target_pt.set_data([tp[0]], [tp[1]])

        speed = state[3] if len(state) == 4 else float(np.hypot(state[3], state[4]))
        ax.set_title(f"{title or result.controller_name}  |  t = {result.t[i]:.1f}s  v = {speed:.1f} m/s")
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
    ax.plot(mpc_result.states[:, 0], mpc_result.states[:, 1], "-", color="#2f6feb", linewidth=2, label=mpc_result.controller_name)
    ax.plot(pp_result.states[:, 0], pp_result.states[:, 1], "-", color="#e8590c", linewidth=2, label=pp_result.controller_name)
    ax.set_aspect("equal")
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title("Driven path vs. reference")
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    ax.plot(mpc_result.t[1:], mpc_result.lateral_error, color="#2f6feb", label=mpc_result.controller_name)
    ax.plot(pp_result.t[1:], pp_result.lateral_error, color="#e8590c", label=pp_result.controller_name)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xlabel("time [s]"); ax.set_ylabel("cross-track error [m]")
    ax.set_title("Lateral (cross-track) tracking error")
    ax.legend(fontsize=9)

    ax = axes[1, 0]
    ax.plot(mpc_result.t[1:], np.rad2deg(mpc_result.controls[:, 1]), color="#2f6feb", label=mpc_result.controller_name)
    ax.plot(pp_result.t[1:], np.rad2deg(pp_result.controls[:, 1]), color="#e8590c", label=pp_result.controller_name)
    ax.set_xlabel("time [s]"); ax.set_ylabel("steering angle [deg]")
    ax.set_title("Control input: steering")
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(mpc_result.t[1:], mpc_result.controls[:, 0], color="#2f6feb", label=mpc_result.controller_name)
    ax.plot(pp_result.t[1:], pp_result.controls[:, 0], color="#e8590c", label=pp_result.controller_name)
    ax.set_xlabel("time [s]"); ax.set_ylabel("acceleration [m/s^2]")
    ax.set_title("Control input: acceleration")
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def obstacle_avoidance_plot(path: np.ndarray, result_with: SimResult, result_without: SimResult,
                             obstacles: list[tuple[float, float, float]], out_path: str):
    """Static before/after figure for the obstacle-avoidance demo: same NMPC
    controller and scenario, avoidance constraint on vs. off.

    Note: deliberately NOT equal-aspect -- the lane is ~100m long with ~2m of
    lateral deviation, so a true equal-aspect plot would render as a nearly
    flat line. The Y axis is visually exaggerated (with axhline lane-edge
    markers to keep it honest about scale) so the maneuver is actually legible.
    """
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Reference lane center", zorder=1)
    ax.plot(result_without.states[:, 0], result_without.states[:, 1], "-", color="#c92a2a",
            linewidth=2.2, label="NMPC, avoidance OFF (collides)", zorder=2)
    ax.plot(result_with.states[:, 0], result_with.states[:, 1], "-", color="#2f6feb",
            linewidth=2.2, label="NMPC, avoidance ON", zorder=3)

    for i, (ox, oy, orad) in enumerate(obstacles):
        ax.add_patch(patches.Circle((ox, oy), orad, facecolor="#f03e3e", edgecolor="#9a1414",
                                     alpha=0.55, zorder=4, label="Obstacle" if i == 0 else None))

    ax.set_xlim(path[:, 0].min() - 3, path[:, 0].max() + 3)
    ax.set_ylim(-3.2, 3.2)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]  (lateral scale exaggerated vs. X -- see note)")
    ax.set_title("Obstacle avoidance: same NMPC controller, hard keep-out constraint on vs. off")
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def moving_obstacle_plot(path: np.ndarray, ped_traj: np.ndarray, results: dict,
                          obstacle_radius: float, out_path: str):
    """Static before/after figure for moving_obstacle_demo.py: the SAME true
    pedestrian trajectory, three ego runs overlaid, one per obstacle-future
    assumption (naive/CV/SSM). `results` maps method name -> the dict
    returned by moving_obstacle_demo.run_scenario."""
    colors = {"naive": "#c92a2a", "cv": "#e8590c", "ssm": "#2f6feb"}
    labels = {"naive": "naive (static)", "cv": "constant-velocity", "ssm": "SSM (learned)"}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Lane centerline", zorder=1)
    ax.plot(ped_traj[:, 0], ped_traj[:, 1], "-", color="#495057", linewidth=2, label="Pedestrian (true path)", zorder=2)
    ax.add_patch(patches.Circle(ped_traj[-1], obstacle_radius, facecolor="#868e96", edgecolor="#343a40",
                                 alpha=0.5, zorder=3, label="Pedestrian, final position"))

    for method in ["naive", "cv", "ssm"]:
        r = results[method]
        tag = " (COLLISION)" if r["collision"] else ""
        ax.plot(r["states"][:, 0], r["states"][:, 1], "-", color=colors[method], linewidth=2.2,
                label=f"Ego, {labels[method]} prediction{tag}", zorder=4)

    ax.set_xlim(path[:, 0].min() - 3, path[:, 0].max() + 3)
    ax.set_ylim(-8, 3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title("Moving-obstacle avoidance: same NMPC, same true pedestrian path,\n"
                 "different assumptions about where the pedestrian will be")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def multimodal_obstacle_plot(path: np.ndarray, ped_traj: np.ndarray, results: dict,
                              obstacle_radius: float, branch: str, out_path: str):
    """Same idea as moving_obstacle_plot, for multimodal_obstacle_demo.py's
    three methods (cv / ssm_uni / ssm_mm) instead of moving_obstacle_demo's
    (naive / cv / ssm) -- kept as a separate function rather than
    parameterizing moving_obstacle_plot so neither demo's plot depends on
    the other's method names. `branch` ("go" or "stop") is the TRUE outcome
    on this particular trial, shown in the title since it's the whole point
    of this demo: the same three methods handling a case the pedestrian
    actually crossed vs. a case they actually stopped."""
    colors = {"cv": "#e8590c", "ssm_uni": "#f08c00", "ssm_mm": "#2f6feb"}
    labels = {"cv": "constant-velocity", "ssm_uni": "unimodal SSM (Part 3)", "ssm_mm": "multimodal SSM (2 hypotheses)"}

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Lane centerline", zorder=1)
    ax.plot(ped_traj[:, 0], ped_traj[:, 1], "-", color="#495057", linewidth=2,
            label=f"Pedestrian (true path -- {branch})", zorder=2)
    ax.add_patch(patches.Circle(ped_traj[-1], obstacle_radius, facecolor="#868e96", edgecolor="#343a40",
                                 alpha=0.5, zorder=3, label="Pedestrian, final position"))

    for method in ["cv", "ssm_uni", "ssm_mm"]:
        r = results[method]
        tag = " (COLLISION)" if r["collision"] else ""
        ax.plot(r["states"][:, 0], r["states"][:, 1], "-", color=colors[method], linewidth=2.2,
                label=f"Ego, {labels[method]}{tag}", zorder=4)

    ax.set_xlim(path[:, 0].min() - 3, path[:, 0].max() + 3)
    ax.set_ylim(-8, 3)
    ax.set_xlabel("X [m]")
    ax.set_ylabel("Y [m]")
    ax.set_title(f"Multimodal obstacle avoidance -- true outcome: pedestrian {branch}\n"
                 "same NMPC, same true path, different obstacle-prediction strategies")
    ax.legend(fontsize=8.5, loc="upper left", bbox_to_anchor=(1.01, 1.0), borderaxespad=0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def animate_moving_obstacle(path: np.ndarray, ped_traj: np.ndarray, result: dict,
                             obstacle_radius: float, out_path: str, title: str = "",
                             fps: int = 20, stride: int = 2):
    """Animated GIF of one ego run (moving_obstacle_demo's `result` dict for
    a single method) against the true, moving pedestrian."""
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Lane centerline", zorder=1)
    ax.set_xlim(path[:, 0].min() - 3, path[:, 0].max() + 3)
    ax.set_ylim(-8, 3)
    ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
    ax.set_title(title or "Moving-obstacle avoidance")

    driven_line, = ax.plot([], [], "-", color="#2f6feb", linewidth=2, label="Ego driven path", zorder=2)
    ped_patch_holder = {"patch": None}
    car_patch_holder = {"patch": None}
    ax.legend(loc="upper right", fontsize=8)

    states = result["states"]
    n_frames = min(len(states), len(ped_traj))
    frames = list(range(0, n_frames, stride))

    def update(frame_idx):
        i = frames[frame_idx]
        state = states[i]
        driven_line.set_data(states[:i + 1, 0], states[:i + 1, 1])

        if car_patch_holder["patch"] is not None:
            car_patch_holder["patch"].remove()
        car_patch_holder["patch"] = _draw_car(ax, state[0], state[1], state[2], color="#2f6feb")

        if ped_patch_holder["patch"] is not None:
            ped_patch_holder["patch"].remove()
        ped_patch_holder["patch"] = ax.add_patch(
            patches.Circle(ped_traj[i], obstacle_radius, facecolor="#868e96", edgecolor="#343a40",
                            alpha=0.7, zorder=3))

        speed = float(np.hypot(state[3], state[4]))
        ax.set_title(f"{title}  |  t = {i * 0.1:.1f}s  v = {speed:.1f} m/s")
        return driven_line,

    anim = FuncAnimation(fig, update, frames=len(frames), interval=1000 / fps, blit=False)
    anim.save(out_path, writer=PillowWriter(fps=fps))
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
