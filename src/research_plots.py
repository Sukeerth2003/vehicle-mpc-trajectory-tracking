"""
Publication-style figures for comparative_study.py's ablation-matrix results:
multi-panel grouped bar charts with error bars, a heatmap/matrix summary, and
solve-time comparisons -- the kind of figure a paper's results section would
have, instead of the single-scenario line/scatter plots visualize.py makes
for the individual Part 3/4 demos.

Reads ../results/comparative_study.json (written by comparative_study.py)
and writes PNGs to ../results/.
"""

from __future__ import annotations

import json

import matplotlib.pyplot as plt
import numpy as np

CONTROLLERS = ["kinematic_ltvmpc", "dynamic_nmpc"]
CONTROLLER_LABELS = {"kinematic_ltvmpc": "Kinematic + LTV-MPC", "dynamic_nmpc": "Dynamic + NMPC"}
METHODS = ["naive", "cv", "ctrv", "ssm_uni", "ssm_mm"]
METHOD_LABELS = {"naive": "Naive", "cv": "CV", "ctrv": "CTRV", "ssm_uni": "SSM\n(unimodal)", "ssm_mm": "SSM\n(multimodal)"}
SCENARIOS = ["moving_stop", "ambiguous_go", "ambiguous_stop"]
SCENARIO_LABELS = {
    "moving_stop": "Scenario A: deterministic\nstopping pedestrian (Part 3)",
    "ambiguous_go": "Scenario B1: ambiguous\npedestrian -- true outcome: go",
    "ambiguous_stop": "Scenario B2: ambiguous\npedestrian -- true outcome: stop",
}
OBSTACLE_RADIUS = 0.6
CONTROLLER_COLORS = {"kinematic_ltvmpc": "#4c72b0", "dynamic_nmpc": "#dd8452"}

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "axes.edgecolor": "#333333",
    "axes.labelcolor": "#222222", "text.color": "#222222", "xtick.color": "#333333",
    "ytick.color": "#333333", "font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
    "figure.titlesize": 14, "figure.titleweight": "bold",
})


def load_results(path="../results/comparative_study.json"):
    with open(path) as f:
        return json.load(f)


def plot_ablation_bars(results: dict, out_path: str):
    """One subplot per scenario: grouped bars (method x controller) of mean
    closest approach with std error bars, and a red dashed line at the
    obstacle's physical radius (below it = a collision by definition)."""
    fig, axes = plt.subplots(1, len(SCENARIOS), figsize=(15, 5), sharey=True)
    x = np.arange(len(METHODS))
    width = 0.35

    for ax, scenario in zip(axes, SCENARIOS):
        for i, controller in enumerate(CONTROLLERS):
            means = [results[scenario][controller][m]["min_dist_mean"] for m in METHODS]
            stds = [results[scenario][controller][m]["min_dist_std"] for m in METHODS]
            n_coll = [results[scenario][controller][m]["n_collisions"] for m in METHODS]
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, means, width, yerr=stds, capsize=3,
                           label=CONTROLLER_LABELS[controller], color=CONTROLLER_COLORS[controller],
                           edgecolor="white", linewidth=0.6)
            for b, c in zip(bars, n_coll):
                if c > 0:
                    ax.annotate(f"{c} coll.", (b.get_x() + b.get_width() / 2, b.get_height() + stds[bars.index(b)] + 0.05),
                                ha="center", fontsize=7.5, color="#c0392b", fontweight="bold")
        ax.axhline(OBSTACLE_RADIUS, color="#c0392b", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_xticks(x)
        ax.set_xticklabels([METHOD_LABELS[m] for m in METHODS], fontsize=8.5)
        ax.set_title(SCENARIO_LABELS[scenario], fontsize=9.5)
        ax.set_ylim(bottom=0)
        ax.grid(axis="y", linestyle=":", alpha=0.4)

    axes[0].set_ylabel("Closest approach, mean ± std (m)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 1.04), frameon=False)
    fig.suptitle("Full ablation matrix: controller x prediction method x scenario", y=1.10)
    fig.text(0.5, -0.02, "Dashed red line = obstacle radius (0.6 m); below it counts as a collision.",
              ha="center", fontsize=8.5, style="italic", color="#555555")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_closest_approach_matrix(results: dict, out_path: str):
    """Heatmap: rows = scenario, columns = (controller, method), color =
    mean closest approach (diverging around the obstacle radius), text =
    value + collision count -- an at-a-glance summary of the whole matrix."""
    cols = [(c, m) for c in CONTROLLERS for m in METHODS]
    data = np.zeros((len(SCENARIOS), len(cols)))
    coll = np.zeros((len(SCENARIOS), len(cols)), dtype=int)
    for i, scenario in enumerate(SCENARIOS):
        for j, (c, m) in enumerate(cols):
            data[i, j] = results[scenario][c][m]["min_dist_mean"]
            coll[i, j] = results[scenario][c][m]["n_collisions"]

    fig, ax = plt.subplots(figsize=(14, 4.5))
    vmax = max(3.0, data.max())
    im = ax.imshow(data, cmap="RdYlGn", vmin=0, vmax=vmax, aspect="auto")
    for i in range(len(SCENARIOS)):
        for j in range(len(cols)):
            txt = f"{data[i, j]:.2f}m"
            if coll[i, j] > 0:
                txt += f"\n({coll[i, j]} coll.)"
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < data[i, j] / vmax < 0.85 else "white")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([f"{CONTROLLER_LABELS[c].split(' + ')[1]}\n{METHOD_LABELS[m].replace(chr(10), ' ')}"
                         for c, m in cols], fontsize=7.5, rotation=0)
    ax.set_yticks(range(len(SCENARIOS)))
    ax.set_yticklabels([SCENARIO_LABELS[s].replace("\n", " ") for s in SCENARIOS], fontsize=8.5)
    ax.set_title("Closest-approach matrix (mean over trials; obstacle radius = 0.6 m)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85)
    cbar.set_label("Closest approach (m)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def plot_solve_time_bars(results: dict, out_path: str):
    """Mean per-step solve time (ms, log scale) -- QP (LTV-MPC) vs. IPOPT
    (NMPC), averaged across scenarios/methods, extending the project's
    existing 'measured, not assumed' solve-time discussion to the obstacle-
    avoidance setting used throughout this ablation study."""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    means, worsts = {}, {}
    for controller in CONTROLLERS:
        vals = [results[s][controller][m]["mean_solve_time_ms"] for s in SCENARIOS for m in METHODS]
        maxvals = [results[s][controller][m]["max_solve_time_ms"] for s in SCENARIOS for m in METHODS]
        means[controller] = np.mean(vals)
        worsts[controller] = np.max(maxvals)

    x = np.arange(len(CONTROLLERS))
    mean_bars = ax.bar(x - 0.15, [means[c] for c in CONTROLLERS], width=0.3, label="Mean solve time",
                        color=[CONTROLLER_COLORS[c] for c in CONTROLLERS])
    worst_bars = ax.bar(x + 0.15, [worsts[c] for c in CONTROLLERS], width=0.3, label="Worst-case solve time",
                         color=[CONTROLLER_COLORS[c] for c in CONTROLLERS], alpha=0.45, hatch="//")
    ax.set_yscale("log")
    ax.set_xticks(x)
    ax.set_xticklabels([CONTROLLER_LABELS[c] for c in CONTROLLERS])
    ax.set_ylabel("Solve time (ms, log scale)")
    ax.set_title("Per-step solve time under obstacle avoidance, averaged across the ablation matrix")
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle=":", alpha=0.4)
    for b in list(mean_bars) + list(worst_bars):
        ax.annotate(f"{b.get_height():.0f}", (b.get_x() + b.get_width() / 2, b.get_height()),
                    ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    results = load_results()
    plot_ablation_bars(results, "../results/ablation_bars.png")
    plot_closest_approach_matrix(results, "../results/ablation_matrix_heatmap.png")
    plot_solve_time_bars(results, "../results/ablation_solve_time.png")
    print("Saved ablation_bars.png, ablation_matrix_heatmap.png, ablation_solve_time.png to ../results/")
