"""
Interactive Streamlit demo: tweak the MPC horizon, cost weights, trajectory
type and target speed, and see the closed-loop result live -- MPC vs. a
pure-pursuit baseline.

Run with:
    streamlit run app.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

from baseline_controller import PurePursuitConfig
from mpc_controller import MPCConfig
from simulate import run_mpc, run_pure_pursuit
from trajectory import TRAJECTORIES, get_trajectory
from vehicle_model import KinematicBicycleModel

st.set_page_config(page_title="MPC Vehicle Trajectory Tracking", layout="wide")

st.title("State-Space Vehicle Model + MPC Trajectory Tracking")
st.markdown(
    "A kinematic bicycle model controlled by **Linear Time-Varying MPC** "
    "(cvxpy/OSQP), benchmarked against a classical **pure pursuit** baseline. "
    "Adjust the controls in the sidebar and re-run the simulation."
)

with st.sidebar:
    st.header("Trajectory")
    traj_name = st.selectbox("Reference path", list(TRAJECTORIES.keys()), index=0)
    v_target = st.slider("Target speed [m/s]", 2.0, 12.0, 5.0 if traj_name == "figure_eight" else 8.0, 0.5)
    sim_time = st.slider("Simulation time [s]", 10.0, 50.0, 30.0, 5.0)

    st.header("MPC settings")
    horizon = st.slider("Prediction horizon N (steps)", 5, 25, 12, 1)
    q_pos = st.slider("Position weight (Q_x, Q_y)", 1.0, 40.0, 15.0, 1.0)
    q_heading = st.slider("Heading weight (Q_psi)", 0.0, 15.0, 3.0, 0.5)
    r_steer = st.slider("Steering effort weight (R_delta)", 0.0, 50.0, 20.0, 1.0)
    rd_steer = st.slider("Steering rate weight (Rd_delta)", 0.0, 80.0, 40.0, 2.0)

    st.header("Vehicle limits")
    max_steer_deg = st.slider("Max steering angle [deg]", 15, 45, 35, 1)
    wheelbase = st.slider("Wheelbase L [m]", 2.0, 3.5, 2.7, 0.1)

    run_button = st.button("Run simulation", type="primary", use_container_width=True)

if run_button or "last_result" not in st.session_state:
    model = KinematicBicycleModel(wheelbase=wheelbase, dt=0.1, max_steer=np.deg2rad(max_steer_deg))

    traj_kwargs = {"v_target": v_target}
    if traj_name == "figure_eight":
        traj_kwargs["scale"] = 30.0
    elif traj_name == "circular":
        traj_kwargs["radius"] = 15.0
    path = get_trajectory(traj_name, **traj_kwargs)
    x0 = np.array([path[0, 0], path[0, 1], path[0, 2], 0.0])

    mpc_cfg = MPCConfig(
        horizon=horizon,
        Q=np.diag([q_pos, q_pos, q_heading, 2.0]),
        Qf=np.diag([q_pos * 1.7, q_pos * 1.7, q_heading * 1.7, 2.0]),
        R=np.diag([1.0, r_steer]),
        Rd=np.diag([5.0, rd_steer]),
    )

    with st.spinner("Solving MPC QPs along the horizon..."):
        mpc_result = run_mpc(path, model, mpc_cfg, sim_time=sim_time, x0=x0.copy())
    with st.spinner("Running pure pursuit baseline..."):
        pp_result = run_pure_pursuit(path, model, PurePursuitConfig(), sim_time=sim_time, x0=x0.copy())

    st.session_state["last_result"] = (path, mpc_result, pp_result)

path, mpc_result, pp_result = st.session_state["last_result"]

col1, col2, col3, col4 = st.columns(4)
col1.metric("MPC mean |cross-track error|", f"{np.mean(np.abs(mpc_result.lateral_error)):.3f} m")
col2.metric("Pure Pursuit mean |cross-track error|", f"{np.mean(np.abs(pp_result.lateral_error)):.3f} m")
col3.metric("MPC max |cross-track error|", f"{np.max(np.abs(mpc_result.lateral_error)):.3f} m")
col4.metric("MPC mean solve time", f"{np.mean(mpc_result.solve_times) * 1000:.1f} ms")

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(path[:, 0], path[:, 1], "--", color="#9aa5b1", linewidth=1.5, label="Reference")
ax.plot(mpc_result.states[:, 0], mpc_result.states[:, 1], color="#2f6feb", linewidth=2, label="MPC")
ax.plot(pp_result.states[:, 0], pp_result.states[:, 1], color="#e8590c", linewidth=2, label="Pure Pursuit")
ax.set_aspect("equal")
ax.set_xlabel("X [m]"); ax.set_ylabel("Y [m]")
ax.legend()
ax.set_title("Driven path vs. reference")

fig2, ax2 = plt.subplots(figsize=(6, 3.5))
ax2.plot(mpc_result.t[1:], mpc_result.lateral_error, color="#2f6feb", label="MPC")
ax2.plot(pp_result.t[1:], pp_result.lateral_error, color="#e8590c", label="Pure Pursuit")
ax2.axhline(0, color="black", linewidth=0.7)
ax2.set_xlabel("time [s]"); ax2.set_ylabel("cross-track error [m]")
ax2.legend()
ax2.set_title("Lateral tracking error over time")

c1, c2 = st.columns(2)
with c1:
    st.pyplot(fig, use_container_width=True)
with c2:
    st.pyplot(fig2, use_container_width=True)

st.caption(
    "Tip: try shrinking the prediction horizon to 5-6 steps, or cranking the "
    "steering rate weight way up, to see MPC's tracking quality degrade -- "
    "that's the trade-off between look-ahead / control smoothness and "
    "tracking accuracy that this whole project is about."
)
