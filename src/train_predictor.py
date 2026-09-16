"""
Train the S4D obstacle-trajectory predictor (ssm_predictor.ObstaclePredictor)
on synthetic motion patterns (obstacle_trajectory_data.py), then validate it
against the two non-learned baselines (trajectory_baselines.py) on held-out
data using ADE/FDE -- the standard trajectory-forecasting metrics -- both
overall and broken down per motion pattern, so an aggregate improvement
can't hide the predictor being no better (or worse) on any one pattern.

Saves the trained weights to ../results/ssm_predictor.pt and a metrics
summary to ../results/ssm_predictor_metrics.json (read by
moving_obstacle_demo.py and cited in the README).
"""

from __future__ import annotations

import json

import numpy as np
import torch
import torch.nn as nn

from obstacle_trajectory_data import PATTERNS, add_observation_noise, build_dataset, split_dataset
from ssm_predictor import ObstaclePredictor, offsets_to_per_step_disp
from trajectory_baselines import ade, fde, predict_ctrv, predict_cv

K, H, DT = 10, 10, 0.1
SENSOR_NOISE_STD = 0.05   # meters -- same order of magnitude as Part 3's sensor noise elsewhere in this project
# Shared with moving_obstacle_demo.py so it constructs an identical model
# before loading ../results/ssm_predictor.pt's state dict.
# Research-scale (Part 5): selective (Mamba-style) S4D blocks, ~330k params --
# up from the original demo-scale plain-S4D model's ~27.6k (d_model=48,
# d_state=12, n_layers=2, selective=False -- still available for comparison).
MODEL_KWARGS = dict(K=K, H=H, d_model=160, d_state=40, n_layers=3, dt=DT, selective=True)


def train(model, train_ds, val_ds, epochs=80, batch_size=64, lr=3e-3, weight_decay=1e-4, seed=0):
    """Sensor noise on the observed window is redrawn fresh every epoch
    (data augmentation) rather than fixed once -- see the note on
    add_observation_noise for why a frozen noisy dataset let the model
    memorize per-sample noise instead of learning to filter it. The
    validation set uses one fixed noise draw (seeded) for a stable,
    reproducible early-stopping signal; the best-val-loss checkpoint is kept
    (validation loss rises after a fairly small number of epochs here, so
    just running to a fixed epoch count and keeping the last checkpoint
    would silently hand back an overfit model)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    loss_fn = nn.MSELoss()

    n = train_ds.past.shape[0]
    val_noisy = add_observation_noise(val_ds, SENSOR_NOISE_STD, np.random.default_rng(12345))
    val_past = torch.tensor(val_noisy.past, dtype=torch.float32)
    val_future = torch.tensor(val_noisy.future, dtype=torch.float32)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        train_noisy = add_observation_noise(train_ds, SENSOR_NOISE_STD, rng)
        past = torch.tensor(train_noisy.past, dtype=torch.float32)
        future = torch.tensor(train_noisy.future, dtype=torch.float32)
        future_disp = offsets_to_per_step_disp(future)

        # Anneal teacher forcing from fully-teacher-forced (epoch 0) to fully
        # autoregressive (last epoch) -- see ssm_predictor.forward's note on
        # exposure bias for why this matters.
        tf_prob = max(0.0, 1.0 - epoch / max(1, epochs - 1))

        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred_offsets, _ = model(past[idx], future_disp[idx], teacher_forcing_prob=tf_prob)
            loss = loss_fn(pred_offsets, future[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred, _ = model(val_past)
            val_loss = loss_fn(val_pred, val_future).item()
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}/{epochs}  tf_prob={tf_prob:.2f}  "
                  f"train_mse={total_loss / n:.5f}  val_mse={val_loss:.5f}")

    print(f"best val_mse={best_val:.5f} at epoch {best_epoch + 1} -- restoring that checkpoint")
    model.load_state_dict(best_state)
    return model


def evaluate(model, ds, dt=DT):
    model.eval()
    past_t = torch.tensor(ds.past, dtype=torch.float32)
    with torch.no_grad():
        ssm_pred, _ = model(past_t)
    ssm_pred = ssm_pred.numpy()

    results = {"ssm": {"ade": [], "fde": []}, "cv": {"ade": [], "fde": []}, "ctrv": {"ade": [], "fde": []}}
    per_pattern = {p: {"ssm": [], "cv": [], "ctrv": []} for p in PATTERNS}

    for i in range(len(ds.past)):
        true_off = ds.future[i]
        cv_pred = predict_cv(ds.past[i], H)
        ctrv_pred = predict_ctrv(ds.past[i], H, dt)

        results["ssm"]["ade"].append(ade(ssm_pred[i], true_off))
        results["ssm"]["fde"].append(fde(ssm_pred[i], true_off))
        results["cv"]["ade"].append(ade(cv_pred, true_off))
        results["cv"]["fde"].append(fde(cv_pred, true_off))
        results["ctrv"]["ade"].append(ade(ctrv_pred, true_off))
        results["ctrv"]["fde"].append(fde(ctrv_pred, true_off))

        per_pattern[ds.patterns[i]]["ssm"].append(ade(ssm_pred[i], true_off))
        per_pattern[ds.patterns[i]]["cv"].append(ade(cv_pred, true_off))
        per_pattern[ds.patterns[i]]["ctrv"].append(ade(ctrv_pred, true_off))

    summary = {method: {"ade_mean": float(np.mean(v["ade"])), "ade_std": float(np.std(v["ade"])),
                         "fde_mean": float(np.mean(v["fde"])), "fde_std": float(np.std(v["fde"]))}
               for method, v in results.items()}
    per_pattern_summary = {p: {m: float(np.mean(vals)) for m, vals in d.items()} for p, d in per_pattern.items()}
    return summary, per_pattern_summary


if __name__ == "__main__":
    import os

    # n_per_pattern/epochs trimmed from the plain-S4D model's 1500/80 to keep
    # the ~12x larger selective-SSM model's CPU training time reasonable
    # (this project trains on 2 CPU cores, no GPU) while still using enough
    # data/steps for the bigger model to converge well past the old model's
    # loss curve -- see the results below for whether that trade-off held up.
    ds = build_dataset(n_per_pattern=800, K=K, H=H, dt=DT, seed=0)
    train_ds, val_ds, test_ds = split_dataset(ds)
    print(f"train={len(train_ds.past)}  val={len(val_ds.past)}  test={len(test_ds.past)}  "
          f"(sensor_noise_std={SENSOR_NOISE_STD} m)")

    model = ObstaclePredictor(**MODEL_KWARGS)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")
    train(model, train_ds, val_ds, epochs=40, lr=2e-3, weight_decay=1e-4)

    test_noisy = add_observation_noise(test_ds, SENSOR_NOISE_STD, np.random.default_rng(999))
    print("\n--- Test set: ADE / FDE (meters), SSM predictor vs. baselines ---")
    summary, per_pattern = evaluate(model, test_noisy)
    for method in ["ssm", "cv", "ctrv"]:
        s = summary[method]
        print(f"{method:6s}  ADE = {s['ade_mean']:.3f} +/- {s['ade_std']:.3f} m   "
              f"FDE = {s['fde_mean']:.3f} +/- {s['fde_std']:.3f} m")

    print("\n--- Per-pattern ADE (meters) ---")
    print(f"{'pattern':20s} {'ssm':>8s} {'cv':>8s} {'ctrv':>8s}")
    for p in PATTERNS:
        d = per_pattern[p]
        print(f"{p:20s} {d['ssm']:8.3f} {d['cv']:8.3f} {d['ctrv']:8.3f}")

    os.makedirs("../results", exist_ok=True)
    torch.save(model.state_dict(), "../results/ssm_predictor.pt")
    with open("../results/ssm_predictor_metrics.json", "w") as f:
        json.dump({"summary": summary, "per_pattern": per_pattern,
                    "K": K, "H": H, "dt": DT}, f, indent=2)
    print("\nSaved ../results/ssm_predictor.pt and ../results/ssm_predictor_metrics.json")
