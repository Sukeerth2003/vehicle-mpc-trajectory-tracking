"""
Renders one of comparative_study.py's recorded trials inside a real 2D
driving-simulator renderer (highway-env, Leurent 2018) instead of this
project's own matplotlib plots, for the "look like a research paper's demo
video" visual the README wants.

Important distinction: highway-env's own vehicle *physics and decision*
models are NOT used here at all -- this script only borrows its road/vehicle
*rendering* (highway_env.road.graphics, highway_env.vehicle.graphics). The
actual trajectories being drawn, frame by frame, are exactly what this
project's own validated NMPC/LTV-MPC controllers and dynamic/kinematic
bicycle models produced in comparative_study.py -- a REPLAY, not a
re-simulation. Re-deriving this project's dynamics and control inside
highway-env's own vehicle/behavior classes would mean re-validating an
entirely different (and less accurate -- highway-env's Vehicle uses a
simpler point-mass-ish kinematic model, no tire slip) physics stack for a
visual-only payoff; replaying already-validated trajectories through a
purpose-built 2D traffic-scene renderer gets the "paper demo video" look
without that trade.

Runs headless (SDL_VIDEODRIVER=dummy) since there is no display in this
environment -- must be set before pygame is imported anywhere, including
transitively via highway_env, hence the very top of this file.
"""

from __future__ import annotations

import os

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("SDL_AUDIODRIVER", "dummy")

import json

import imageio.v2 as imageio
import numpy as np
import pygame
from highway_env.road.graphics import RoadGraphics, WorldSurface
from highway_env.road.lane import LineType, StraightLane
from highway_env.road.road import Road, RoadNetwork
from highway_env.vehicle.graphics import VehicleGraphics
from highway_env.vehicle.kinematics import Vehicle

LANE_LENGTH = 100.0
LANE_WIDTH = 4.0
OBSTACLE_RADIUS = 0.6
FRAME_SIZE = (760, 340)
SCALING = 11.0

# Distinct, colorblind-legible colors for the multi-method comparison scene,
# one ego vehicle per prediction method.
METHOD_COLORS = {
    "naive": (127, 127, 127),
    "cv": (76, 114, 176),
    "ctrv": (85, 168, 104),
    "ssm_uni": (221, 132, 82),
    "ssm_mm": (196, 78, 82),
}
METHOD_LABELS = {
    "naive": "Naive", "cv": "CV", "ctrv": "CTRV",
    "ssm_uni": "SSM (unimodal)", "ssm_mm": "SSM (multimodal)",
}

# Layout for the stacked multi-method comparison GIF: one mini-panel per
# prediction method, each with its own camera (centered on its own ego, same
# as render_trial), stacked vertically with a small header per panel plus one
# title bar for the whole figure.
PANEL_SIZE = (760, 150)
PANEL_SCALING = 9.0
PANEL_HEADER_H = 24
TITLE_BAR_H = 28


def build_road():
    net = RoadNetwork()
    lane = StraightLane([0, 0], [LANE_LENGTH, 0], width=LANE_WIDTH,
                         line_types=(LineType.CONTINUOUS, LineType.CONTINUOUS))
    net.add_lane("a", "b", lane)
    return Road(network=net, np_random=np.random.RandomState(0), record_history=False)


def render_trial(ped_traj: np.ndarray, ego_states: np.ndarray, out_path: str,
                  title: str = "", stride: int = 2, fps: int = 12):
    """ped_traj: (T, 2) true pedestrian [x, y]. ego_states: (T, >=3)
    [x, y, psi, ...] from comparative_study's recorded `states`. Renders
    every `stride`-th step to keep GIF size/frame count reasonable."""
    pygame.init()
    road = build_road()

    ego = Vehicle(road, position=[ego_states[0, 0], ego_states[0, 1]], heading=ego_states[0, 2], speed=8.0)
    ego.color = VehicleGraphics.EGO_COLOR
    ped = Vehicle(road, position=[ped_traj[0, 0], ped_traj[0, 1]], heading=np.pi / 2, speed=1.5)
    ped.LENGTH, ped.WIDTH = OBSTACLE_RADIUS * 2, OBSTACLE_RADIUS * 2
    ped.color = (220, 30, 30)
    road.vehicles = [ego, ped]

    n = min(len(ped_traj), len(ego_states))
    frames = []
    for t in range(0, n, stride):
        ego.position = np.array([ego_states[t, 0], ego_states[t, 1]])
        ego.heading = float(ego_states[t, 2])
        ped.position = np.array([ped_traj[t, 0], ped_traj[t, 1]])

        surf = WorldSurface(FRAME_SIZE, 0, pygame.Surface(FRAME_SIZE))
        surf.scaling = SCALING
        surf.centering_position = [0.4, 0.5]
        surf.move_display_window_to(ego.position)
        RoadGraphics.display(road, surf)
        RoadGraphics.display_traffic(road, surf, offscreen=True)
        # Overlay the pedestrian's hard keep-out radius (what the NMPC/QP
        # constraint actually enforces) as a translucent ring, so the safety
        # margin the controller is reasoning about is visible, not implicit.
        ring = pygame.Surface(FRAME_SIZE, pygame.SRCALPHA)
        ped_px = surf.pos2pix(ped.position[0], ped.position[1])
        pygame.draw.circle(ring, (220, 30, 30, 70), ped_px, int(OBSTACLE_RADIUS * SCALING))
        surf.blit(ring, (0, 0))

        frame = pygame.surfarray.array3d(surf)
        frame = np.transpose(frame, (1, 0, 2))   # pygame (W,H,C) -> imageio (H,W,C)
        frames.append(frame)

    imageio.mimsave(out_path, frames, fps=fps, loop=0)
    return out_path


def _panel_frames(ped_traj: np.ndarray, ego_states: np.ndarray, color: tuple,
                   n_steps: int, stride: int) -> list[np.ndarray]:
    """Renders ONE method's own dedicated scene (its own Road/ego/pedestrian,
    its own camera centered on its own ego every frame -- exactly the
    render_trial recipe, which is the version already proven not to lose
    vehicles off-frame). Returns raw (H, W, C) arrays, no header baked in."""
    pygame.init()
    road = build_road()
    ego = Vehicle(road, position=[ego_states[0, 0], ego_states[0, 1]], heading=ego_states[0, 2], speed=8.0)
    ego.color = color
    ped = Vehicle(road, position=[ped_traj[0, 0], ped_traj[0, 1]], heading=np.pi / 2, speed=1.5)
    ped.LENGTH, ped.WIDTH = OBSTACLE_RADIUS * 2, OBSTACLE_RADIUS * 2
    ped.color = (20, 20, 20)
    road.vehicles = [ego, ped]

    frames = []
    for t in range(0, n_steps, stride):
        ego.position = np.array([ego_states[t, 0], ego_states[t, 1]])
        ego.heading = float(ego_states[t, 2])
        ped.position = np.array([ped_traj[t, 0], ped_traj[t, 1]])

        surf = WorldSurface(PANEL_SIZE, 0, pygame.Surface(PANEL_SIZE))
        surf.scaling = PANEL_SCALING
        surf.centering_position = [0.4, 0.5]
        surf.move_display_window_to(ego.position)
        RoadGraphics.display(road, surf)
        RoadGraphics.display_traffic(road, surf, offscreen=True)

        ring = pygame.Surface(PANEL_SIZE, pygame.SRCALPHA)
        ped_px = surf.pos2pix(ped.position[0], ped.position[1])
        pygame.draw.circle(ring, (20, 20, 20, 80), ped_px, int(OBSTACLE_RADIUS * PANEL_SCALING))
        surf.blit(ring, (0, 0))

        frame = pygame.surfarray.array3d(surf)
        frame = np.transpose(frame, (1, 0, 2))   # pygame (W,H,C) -> imageio (H,W,C)
        frames.append(frame)
    return frames


def _header_bar(label: str, color: tuple, collided: bool, width: int, height: int, font) -> np.ndarray:
    """A static (doesn't change frame to frame) header strip: colored swatch
    + method name, red 'COLLIDED' tag if this method collided on this trial
    (ground truth from comparative_study.json's collision_trials, not
    eyeballed off the render)."""
    surf = pygame.Surface((width, height))
    surf.fill((245, 245, 245) if not collided else (253, 233, 231))
    pygame.draw.rect(surf, color, pygame.Rect(8, height // 2 - 7, 14, 14))
    text = label + ("   COLLIDED" if collided else "")
    text_color = (192, 57, 43) if collided else (20, 20, 20)
    txt = font.render(text, True, text_color)
    surf.blit(txt, (28, height // 2 - txt.get_height() // 2))
    pygame.draw.line(surf, (180, 180, 180), (0, height - 1), (width, height - 1), 1)
    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2))


def _title_bar(title: str, width: int, height: int, font) -> np.ndarray:
    surf = pygame.Surface((width, height))
    surf.fill((30, 30, 30))
    txt = font.render(title, True, (255, 255, 255))
    surf.blit(txt, (10, height // 2 - txt.get_height() // 2))
    arr = pygame.surfarray.array3d(surf)
    return np.transpose(arr, (1, 0, 2))


def render_stacked_comparison(scenario: str, controller: str, trial_idx: int, methods: list[str],
                               trials: dict, collided: dict, out_path: str,
                               stride: int = 2, fps: int = 12):
    """One GIF per (scenario, controller, trial): a vertical STACK of one
    mini-panel per prediction method, each panel its own independent scene
    (own camera, own road) rather than all 5 egos sharing one scene.

    This replaces an earlier version that drew all 5 egos together in one
    shared highway-env scene. That approach had two real, user-caught
    problems: (1) egos that pulled far apart from each other could end up
    outside the shared camera's frame (the camera followed the *mean*
    ego position, so a method that continued driving while others stopped
    could exit the frame entirely -- "not all 5 cars there"), and (2) with
    several rotated, overlapping vehicle rectangles drawn in the same small
    area, which color ended up on top visually shifted frame to frame as the
    vehicles rotated/moved relative to each other -- not an actual color
    change in the data, but a real, confusing rendering artifact ("their
    colour is changing"). Giving every method its own dedicated panel (same
    per-ego camera-follow recipe as the single-method render_trial, already
    proven not to lose the vehicle) removes both problems outright."""
    pygame.init()
    font_header = pygame.font.Font(None, 20)
    font_title = pygame.font.Font(None, 22)

    panel_w, panel_h = PANEL_SIZE
    n_min = None
    per_method = {}
    for method in methods:
        key = f"{scenario}|{controller}|{method}|{trial_idx}"
        if key not in trials:
            print(f"  skip method {method}: {key} not in comparative_study_trials.json")
            continue
        data = trials[key]
        ped_traj = np.array(data["ped_traj"])
        states = np.array(data["states"])[1:]
        n = min(len(ped_traj), len(states))
        n_min = n if n_min is None else min(n_min, n)
        per_method[method] = (ped_traj, states)

    if not per_method:
        print(f"  no methods found for {scenario}|{controller}|trial {trial_idx}, skipping")
        return None

    title = f"{scenario} | {controller} | trial {trial_idx}"
    total_w = panel_w
    n_frames = len(range(0, n_min, stride))
    title_img = _title_bar(title, total_w, TITLE_BAR_H, font_title)

    stacked_frames = [np.tile(title_img, (1, 1, 1)) for _ in range(n_frames)]
    # Build each method's frames (panel scene + its static header, stacked)
    # then append to the running per-frame composite so the final image is
    # title bar, then panel 1, panel 2, ... panel 5, in a single column.
    for method in methods:
        if method not in per_method:
            continue
        ped_traj, states = per_method[method]
        color = METHOD_COLORS.get(method, VehicleGraphics.EGO_COLOR)
        label = METHOD_LABELS.get(method, method)
        header = _header_bar(label, color, bool(collided.get(method)), panel_w, PANEL_HEADER_H, font_header)
        scene_frames = _panel_frames(ped_traj, states, color, n_min, stride)
        for i, scene in enumerate(scene_frames):
            panel = np.concatenate([header, scene], axis=0)
            stacked_frames[i] = np.concatenate([stacked_frames[i], panel], axis=0)

    imageio.mimsave(out_path, stacked_frames, fps=fps, loop=0)
    return out_path


if __name__ == "__main__":
    with open("../results/comparative_study_trials.json") as f:
        trials = json.load(f)
    with open("../results/comparative_study.json") as f:
        study = json.load(f)

    # Flagship visual: the safety-critical "stop" branch, dynamic+NMPC with
    # the multimodal SSM -- the headline single-method result (see README).
    targets = [
        ("ambiguous_stop|dynamic_nmpc|ssm_mm", "../results/highwayenv_ambiguous_stop_mm.gif"),
        ("ambiguous_stop|dynamic_nmpc|ssm_uni", "../results/highwayenv_ambiguous_stop_uni.gif"),
        ("moving_stop|dynamic_nmpc|ssm_uni", "../results/highwayenv_moving_stop.gif"),
    ]
    for key, out_path in targets:
        base_key = "|".join(key.split("|")[:3]) + "|0"   # trial 0 for the single-method clips
        if base_key not in trials:
            print(f"skip {key}: not found in comparative_study_trials.json")
            continue
        data = trials[base_key]
        ped_traj = np.array(data["ped_traj"])
        states = np.array(data["states"])[1:]   # drop the initial pre-step state to align with ped_traj[0]
        render_trial(ped_traj, states, out_path)
        print(f"saved {out_path}")

    # Multi-method comparison scenes: all 5 prediction methods replayed, one
    # per stacked panel, against the identical true trajectory, for every
    # (scenario, controller) pair the ablation study covers -- not a
    # cherry-picked subset, so nothing is hidden either way. Trial indices
    # are chosen to be illustrative: trial 1 for the two (scenario,
    # controller) cells that actually produced a kinematic+LTV-MPC collision
    # (see comparative_study.json's collision_trials), trial 0 (the default)
    # everywhere else.
    all_methods = ["naive", "cv", "ctrv", "ssm_uni", "ssm_mm"]
    comparison_targets = [
        # (scenario, controller, trial_idx, out_path)
        ("moving_stop", "kinematic_ltvmpc", 1, "../results/highwayenv_stack_moving_stop_kinematic.gif"),
        ("moving_stop", "dynamic_nmpc", 0, "../results/highwayenv_stack_moving_stop_dynamic.gif"),
        ("ambiguous_go", "kinematic_ltvmpc", 1, "../results/highwayenv_stack_ambiguous_go_kinematic.gif"),
        ("ambiguous_go", "dynamic_nmpc", 0, "../results/highwayenv_stack_ambiguous_go_dynamic.gif"),
        ("ambiguous_stop", "kinematic_ltvmpc", 0, "../results/highwayenv_stack_ambiguous_stop_kinematic.gif"),
        ("ambiguous_stop", "dynamic_nmpc", 0, "../results/highwayenv_stack_ambiguous_stop_dynamic.gif"),
    ]
    for scenario, controller, trial_idx, out_path in comparison_targets:
        collided = {}
        for method in all_methods:
            try:
                flags = study[scenario][controller][method]["collision_trials"]
                collided[method] = bool(flags[trial_idx])
            except (KeyError, IndexError):
                collided[method] = False
        print(f"rendering stack: {scenario}|{controller}|trial {trial_idx} -> {out_path}")
        result = render_stacked_comparison(scenario, controller, trial_idx, all_methods, trials, collided, out_path)
        if result:
            print(f"saved {out_path}  (collided: {[m for m in all_methods if collided.get(m)]})")
