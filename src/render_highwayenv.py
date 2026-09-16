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


if __name__ == "__main__":
    with open("../results/comparative_study_trials.json") as f:
        trials = json.load(f)

    # Flagship visual: the safety-critical "stop" branch, dynamic+NMPC with
    # the multimodal SSM -- the headline result (see the README).
    targets = [
        ("ambiguous_stop|dynamic_nmpc|ssm_mm", "../results/highwayenv_ambiguous_stop_mm.gif"),
        ("ambiguous_stop|dynamic_nmpc|ssm_uni", "../results/highwayenv_ambiguous_stop_uni.gif"),
        ("moving_stop|dynamic_nmpc|ssm_uni", "../results/highwayenv_moving_stop.gif"),
    ]
    for key, out_path in targets:
        if key not in trials:
            print(f"skip {key}: not found in comparative_study_trials.json")
            continue
        data = trials[key]
        ped_traj = np.array(data["ped_traj"])
        states = np.array(data["states"])[1:]   # drop the initial pre-step state to align with ped_traj[0]
        render_trial(ped_traj, states, out_path)
        print(f"saved {out_path}")
