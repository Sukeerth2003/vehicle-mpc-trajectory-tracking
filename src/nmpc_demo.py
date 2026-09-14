"""
Part 2 headline demo: NMPC (dynamic bicycle model, tire slip) vs. pure pursuit
on the figure-eight, mirroring what visualize.py's __main__ does for Part 1's
LTV-MPC. Produces nmpc_comparison_plot.png and nmpc_tracking.gif in results/.
"""

import os

import numpy as np

from baseline_controller import PurePursuitConfig
from dynamic_vehicle_model import DynamicBicycleModel, VehicleParams
from nmpc_controller import NMPCConfig
from simulate import run_nmpc, run_pure_pursuit
from trajectory import get_trajectory
from visualize import animate_tracking, comparison_plot

if __name__ == "__main__":
    os.makedirs("../results", exist_ok=True)
    model = DynamicBicycleModel(params=VehicleParams(), dt=0.1)
    path = get_trajectory("figure_eight", v_target=5.0, scale=30.0)
    x0 = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0, 0.0, 0.0])

    nmpc_result = run_nmpc(path, model, NMPCConfig(horizon=10), sim_time=40.0, x0=x0.copy())
    pp_result = run_pure_pursuit(path, model, PurePursuitConfig(), sim_time=40.0, x0=x0.copy())

    print(f"NMPC : mean|err|={np.mean(np.abs(nmpc_result.lateral_error)):.3f} m, "
          f"max={np.max(np.abs(nmpc_result.lateral_error)):.3f} m")
    print(f"PP   : mean|err|={np.mean(np.abs(pp_result.lateral_error)):.3f} m, "
          f"max={np.max(np.abs(pp_result.lateral_error)):.3f} m")

    comparison_plot(path, nmpc_result, pp_result, "../results/nmpc_comparison_plot.png")
    animate_tracking(path, nmpc_result, "../results/nmpc_tracking.gif",
                      title="NMPC (dynamic model) tracking a figure-eight", stride=3)
    print("Saved nmpc_comparison_plot.png and nmpc_tracking.gif to ../results/")
