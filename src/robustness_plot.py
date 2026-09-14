"""Bar chart (with per-trial scatter) of the robustness study results in
../results/robustness_summary.json -- run robustness_experiment.py first."""

import json

import matplotlib.pyplot as plt
import numpy as np

LABELS = {
    "ltv_mpc": "LTV-MPC\n(kinematic model)",
    "nmpc": "NMPC\n(dynamic model)",
    "pure_pursuit_kinematic": "Pure Pursuit\n(kinematic model)",
    "pure_pursuit_dynamic": "Pure Pursuit\n(dynamic model)",
}
COLORS = {
    "ltv_mpc": "#2f6feb",
    "nmpc": "#0ca678",
    "pure_pursuit_kinematic": "#e8590c",
    "pure_pursuit_dynamic": "#f08c00",
}

if __name__ == "__main__":
    with open("../results/robustness_summary.json") as f:
        data = json.load(f)
    results, summary = data["results"], data["summary"]
    names = list(LABELS.keys())

    fig, ax = plt.subplots(figsize=(8, 5.5))
    means = [summary[n][0] for n in names]
    stds = [summary[n][1] for n in names]
    x = np.arange(len(names))

    ax.bar(x, means, yerr=stds, capsize=5, color=[COLORS[n] for n in names], alpha=0.85, zorder=2)
    for i, n in enumerate(names):
        trials = results[n]
        jitter = (np.random.default_rng(0).random(len(trials)) - 0.5) * 0.25
        ax.scatter(x[i] + jitter, trials, color="black", s=18, zorder=3, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels([LABELS[n] for n in names], fontsize=9)
    ax.set_ylabel("mean |cross-track error| per trial [m]")
    ax.set_title("Robustness under crosswind + sensor/process noise\n"
                  "(double lane change, 6 trials, error bars = 1 std)")
    fig.tight_layout()
    fig.savefig("../results/robustness_plot.png", dpi=150)
    print("Saved ../results/robustness_plot.png")
