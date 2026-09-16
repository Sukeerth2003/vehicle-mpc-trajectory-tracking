"""
Train the multimodal (mixture-of-trajectories) SSM predictor
(ssm_predictor.MultimodalObstaclePredictor) and check, honestly, whether it
actually learned to be multimodal -- not just whether its average error
looks good.

Getting real mode specialization out of a winner-take-all mixture turned out
to be the hard part of this extension, and it's worth being upfront about
what did and didn't work rather than only showing the final recipe:

  1. Plain winner-take-all (mtp_loss with no `forced_winner`, argmin
     deciding which mode is "responsible" for each sample) collapsed: both
     modes converged to nearly the same output regardless of the true
     branch. Checked directly (not just inferred) by looking at each mode's
     *predicted final displacement* on held-out "branch" samples, split by
     true outcome -- both modes tracked close to the "go" (keep-moving)
     behavior even on samples that were actually "stop", meaning the mixture
     wasn't representing the stop hypothesis *at all*, not just imperfectly.
  2. The cross-entropy term that trains the mode-probability head was
     actively making this worse: it backpropagated into the shared encoder,
     which then got pulled around trying to extract a signal that
     mathematically isn't in the input (see obstacle_trajectory_data.py's
     "branch" pattern -- the observed window is identical in distribution
     regardless of outcome), corrupting the otherwise-learnable decoder
     specialization sitting on top of it. Fixed by detaching the encoder's
     output before the mode-probability head (see
     ssm_predictor.MultimodalObstaclePredictor.forward) -- the classifier
     becomes a pure readout that can't damage the shared representation.
  3. Even after that fix, unsupervised winner-take-all on "branch" samples
     stayed close to 50/50 between the two modes with no consistent
     specialization (the classic rich-get-richer instability -- see
     ssm_predictor.mtp_loss's docstring). The fix actually used here is
     more direct: since this is synthetic data, the true go/stop label for
     each "branch" sample is known at training time (from the generator),
     even though the model is never given it as input. Passing that label
     in as `forced_winner` -- so mode 0 always gets the regression gradient
     for "go" samples and mode 1 always gets it for "stop" samples --
     sidesteps the discovery problem entirely. The model still can't see
     the label at inference time, so this is training supervision, not
     input leakage; it is exactly analogous to how a real multi-agent
     dataset with logged outcomes would be used.

Even with both fixes, specialization is real but incomplete: mode 1 learns
to predict *less* forward displacement than mode 0 on ambiguous samples, but
doesn't fully converge to the true "quick full stop" magnitude within this
training budget. See the branch diagnostics printed below and the README for
the actual numbers and what that does and doesn't mean for the downstream
NMPC demo.

Saves ../results/ssm_predictor_multimodal.pt and
../results/ssm_predictor_multimodal_metrics.json (read by
multimodal_obstacle_demo.py and cited in the README).
"""

from __future__ import annotations

import json

import numpy as np
import torch

from obstacle_trajectory_data import (
    AMBIGUOUS_PATTERN, ObstacleDataset, PATTERNS, add_observation_noise, build_dataset, split_dataset,
)
from ssm_predictor import MultimodalObstaclePredictor, mtp_loss
from trajectory_baselines import ade, fde, predict_ctrv, predict_cv

K, H, DT = 10, 10, 0.1
N_MODES = 2   # matches the true structure of the "branch" pattern (go / stop) -- see the README
SENSOR_NOISE_STD = 0.05
# Shared with multimodal_obstacle_demo.py so it constructs an identical
# model before loading ../results/ssm_predictor_multimodal.pt's state dict.
# Research-scale (Part 5): selective (Mamba-style) blocks, ~212k params, up
# from the original demo-scale model's much smaller plain-S4D encoder.
MODEL_KWARGS = dict(K=K, H=H, d_model=128, d_state=32, n_layers=3, dt=DT, n_modes=N_MODES, selective=True)


def build_training_dataset(n_unimodal: int = 500, n_branch: int = 800, seed: int = 0) -> ObstacleDataset:
    """The four unimodal patterns at `n_unimodal` each, PLUS "branch" (the
    ambiguous pattern -- see obstacle_trajectory_data.py) oversampled to
    `n_branch`, so the one pattern that actually needs mode specialization
    isn't diluted to a 1-in-5 share of the gradient signal."""
    ds_unimodal = build_dataset(n_per_pattern=n_unimodal, K=K, H=H, dt=DT, seed=seed, patterns=PATTERNS)
    ds_branch = build_dataset(n_per_pattern=n_branch, K=K, H=H, dt=DT, seed=seed + 1, patterns=[AMBIGUOUS_PATTERN])
    ds = ObstacleDataset(
        past_clean=np.concatenate([ds_unimodal.past_clean, ds_branch.past_clean]),
        past=np.concatenate([ds_unimodal.past, ds_branch.past]),
        future=np.concatenate([ds_unimodal.future, ds_branch.future]),
        patterns=ds_unimodal.patterns + ds_branch.patterns,
        K=K, H=H, dt=DT,
        branches=ds_unimodal.branches + ds_branch.branches,
    )
    idx = np.random.default_rng(seed + 2).permutation(len(ds.past))
    return ObstacleDataset(
        past_clean=ds.past_clean[idx], past=ds.past[idx], future=ds.future[idx],
        patterns=[ds.patterns[i] for i in idx], K=K, H=H, dt=DT,
        branches=[ds.branches[i] for i in idx],
    )


def _forced_winner_for(patterns: list, branches: list) -> torch.Tensor:
    """-1 (use ordinary arg-min competition) for every non-"branch" sample;
    0 or 1 (the known true go/stop label) for "branch" samples -- see the
    module docstring for why this privileged-at-training-time label is used
    instead of letting the mixture discover the split unsupervised."""
    fw = torch.full((len(patterns),), -1, dtype=torch.long)
    for i, (p, b) in enumerate(zip(patterns, branches)):
        if p == AMBIGUOUS_PATTERN:
            fw[i] = 0 if b == "go" else 1
    return fw


def train(model, train_ds, val_ds, epochs=50, batch_size=64, lr=2e-3, weight_decay=1e-4,
          cls_weight=1.0, seed=0):
    """No teacher-forcing schedule here (unlike Part 3's train_predictor.py)
    -- MultimodalObstaclePredictor is always fully autoregressive, by
    construction (see its docstring), so there's no exposure-bias gap to
    anneal away. Sensor noise is still redrawn fresh every epoch, for the
    same data-augmentation reason as Part 3. See the module docstring for
    what `forced_winner` (computed fresh each epoch, since add_observation_
    noise doesn't change which samples are "branch") is doing and why it was
    necessary."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed + 1)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    n = train_ds.past.shape[0]
    val_noisy = add_observation_noise(val_ds, SENSOR_NOISE_STD, np.random.default_rng(12345))
    val_past = torch.tensor(val_noisy.past, dtype=torch.float32)
    val_future = torch.tensor(val_noisy.future, dtype=torch.float32)
    val_forced = _forced_winner_for(val_noisy.patterns, val_noisy.branches)

    best_val, best_state, best_epoch = float("inf"), None, -1
    for epoch in range(epochs):
        model.train()
        train_noisy = add_observation_noise(train_ds, SENSOR_NOISE_STD, rng)
        past = torch.tensor(train_noisy.past, dtype=torch.float32)
        future = torch.tensor(train_noisy.future, dtype=torch.float32)
        forced = _forced_winner_for(train_noisy.patterns, train_noisy.branches)

        perm = torch.randperm(n)
        total_loss = 0.0
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            opt.zero_grad()
            pred_offsets, _, mode_logits = model(past[idx])
            loss, _, _ = mtp_loss(pred_offsets, mode_logits, future[idx],
                                   cls_weight=cls_weight, forced_winner=forced[idx])
            loss.backward()
            opt.step()
            total_loss += loss.item() * len(idx)
        sched.step()

        model.eval()
        with torch.no_grad():
            val_pred, _, val_logits = model(val_past)
            val_loss, _, _ = mtp_loss(val_pred, val_logits, val_future,
                                       cls_weight=cls_weight, forced_winner=val_forced)
            val_loss = val_loss.item()
        if val_loss < best_val:
            best_val, best_epoch = val_loss, epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"epoch {epoch + 1:3d}/{epochs}  train_mtp_loss={total_loss / n:.5f}  "
                  f"val_mtp_loss={val_loss:.5f}")

    print(f"best val_mtp_loss={best_val:.5f} at epoch {best_epoch + 1} -- restoring that checkpoint")
    model.load_state_dict(best_state)
    return model


def evaluate(model, ds):
    """minADE_M/minFDE_M (best-of-M) overall and per-pattern, plus the
    branch-specific specialization diagnostics: mode usage (which mode won,
    per true branch) AND -- the more direct, harder-to-fake check -- each
    mode's mean predicted *final displacement* split by true branch. Usage
    alone can look reasonable even when both modes predict nearly the same
    thing (see the module docstring); comparing actual predicted magnitudes
    against the true go/stop magnitudes is what actually answers "did this
    learn two different behaviors." Returns (summary, per_pattern,
    branch_diagnostics)."""
    model.eval()
    past_t = torch.tensor(ds.past, dtype=torch.float32)
    with torch.no_grad():
        pred_offsets, _, mode_logits = model(past_t)   # (N, M, H, 2), (N, M)
    pred_offsets = pred_offsets.numpy()
    mode_probs = torch.softmax(mode_logits, dim=-1).numpy()   # (N, M)
    n_modes = pred_offsets.shape[1]
    final_disp = np.linalg.norm(pred_offsets[:, :, -1, :], axis=-1)   # (N, M)

    def min_ade_fde(preds_mh2, true_h2):
        ades = [ade(preds_mh2[m], true_h2) for m in range(preds_mh2.shape[0])]
        fdes = [fde(preds_mh2[m], true_h2) for m in range(preds_mh2.shape[0])]
        best = int(np.argmin(ades))
        return ades[best], fdes[best], best

    all_patterns = sorted(set(ds.patterns))
    results = {"ssm_mm": {"ade": [], "fde": []}, "cv": {"ade": [], "fde": []}, "ctrv": {"ade": [], "fde": []}}
    per_pattern = {p: {"ssm_mm": [], "cv": [], "ctrv": []} for p in all_patterns}

    winner_by_branch = {"go": [], "stop": []}
    final_disp_by_branch = {"go": {"mode0": [], "mode1": []}, "stop": {"mode0": [], "mode1": []}}
    true_final_disp_by_branch = {"go": [], "stop": []}
    top_prob_by_group = {"branch": [], "unambiguous": []}

    for i in range(len(ds.past)):
        true_off = ds.future[i]
        mm_ade, mm_fde, winner = min_ade_fde(pred_offsets[i], true_off)
        cv_pred = predict_cv(ds.past[i], H)
        ctrv_pred = predict_ctrv(ds.past[i], H, DT)

        results["ssm_mm"]["ade"].append(mm_ade)
        results["ssm_mm"]["fde"].append(mm_fde)
        results["cv"]["ade"].append(ade(cv_pred, true_off))
        results["cv"]["fde"].append(fde(cv_pred, true_off))
        results["ctrv"]["ade"].append(ade(ctrv_pred, true_off))
        results["ctrv"]["fde"].append(fde(ctrv_pred, true_off))

        per_pattern[ds.patterns[i]]["ssm_mm"].append(mm_ade)
        per_pattern[ds.patterns[i]]["cv"].append(ade(cv_pred, true_off))
        per_pattern[ds.patterns[i]]["ctrv"].append(ade(ctrv_pred, true_off))

        top_prob = float(mode_probs[i].max())
        if ds.patterns[i] == AMBIGUOUS_PATTERN:
            top_prob_by_group["branch"].append(top_prob)
            branch = ds.branches[i]
            if branch in ("go", "stop"):
                winner_by_branch[branch].append(winner)
                true_final_disp_by_branch[branch].append(float(np.linalg.norm(true_off[-1])))
                final_disp_by_branch[branch]["mode0"].append(float(final_disp[i, 0]))
                final_disp_by_branch[branch]["mode1"].append(float(final_disp[i, 1]))
        elif ds.patterns[i] == "constant_velocity":
            top_prob_by_group["unambiguous"].append(top_prob)

    summary = {method: {"ade_mean": float(np.mean(v["ade"])), "ade_std": float(np.std(v["ade"])),
                         "fde_mean": float(np.mean(v["fde"])), "fde_std": float(np.std(v["fde"]))}
               for method, v in results.items()}
    per_pattern_summary = {p: {m: float(np.mean(vals)) for m, vals in d.items()} for p, d in per_pattern.items()}

    def usage_hist(idx_list):
        if not idx_list:
            return {}
        counts = np.bincount(idx_list, minlength=n_modes)
        return {str(m): float(counts[m] / len(idx_list)) for m in range(n_modes)}

    branch_diagnostics = {
        "mode_usage_go": usage_hist(winner_by_branch["go"]),
        "mode_usage_stop": usage_hist(winner_by_branch["stop"]),
        "n_go": len(winner_by_branch["go"]),
        "n_stop": len(winner_by_branch["stop"]),
        "mean_final_disp": {
            "true_go": float(np.mean(true_final_disp_by_branch["go"])) if true_final_disp_by_branch["go"] else None,
            "true_stop": float(np.mean(true_final_disp_by_branch["stop"])) if true_final_disp_by_branch["stop"] else None,
            "mode0_on_go": float(np.mean(final_disp_by_branch["go"]["mode0"])) if final_disp_by_branch["go"]["mode0"] else None,
            "mode0_on_stop": float(np.mean(final_disp_by_branch["stop"]["mode0"])) if final_disp_by_branch["stop"]["mode0"] else None,
            "mode1_on_go": float(np.mean(final_disp_by_branch["go"]["mode1"])) if final_disp_by_branch["go"]["mode1"] else None,
            "mode1_on_stop": float(np.mean(final_disp_by_branch["stop"]["mode1"])) if final_disp_by_branch["stop"]["mode1"] else None,
        },
        "mean_top_mode_prob_branch": float(np.mean(top_prob_by_group["branch"])) if top_prob_by_group["branch"] else None,
        "mean_top_mode_prob_unambiguous": float(np.mean(top_prob_by_group["unambiguous"])) if top_prob_by_group["unambiguous"] else None,
    }
    return summary, per_pattern_summary, branch_diagnostics


if __name__ == "__main__":
    import os

    ds = build_training_dataset(n_unimodal=500, n_branch=800, seed=0)
    train_ds, val_ds, test_ds = split_dataset(ds)
    print(f"train={len(train_ds.past)}  val={len(val_ds.past)}  test={len(test_ds.past)}  "
          f"(sensor_noise_std={SENSOR_NOISE_STD} m, branch pattern oversampled to help specialization)")

    model = MultimodalObstaclePredictor(**MODEL_KWARGS)
    print(f"model params: {sum(p.numel() for p in model.parameters())}")
    train(model, train_ds, val_ds, epochs=30)

    test_noisy = add_observation_noise(test_ds, SENSOR_NOISE_STD, np.random.default_rng(999))
    print("\n--- Test set: minADE_M / minFDE_M (meters), multimodal SSM vs. baselines ---")
    summary, per_pattern, branch_diag = evaluate(model, test_noisy)
    for method in ["ssm_mm", "cv", "ctrv"]:
        s = summary[method]
        print(f"{method:8s} ADE = {s['ade_mean']:.3f} +/- {s['ade_std']:.3f} m   "
              f"FDE = {s['fde_mean']:.3f} +/- {s['fde_std']:.3f} m")

    print("\n--- Per-pattern minADE (meters) ---")
    print(f"{'pattern':20s} {'ssm_mm':>8s} {'cv':>8s} {'ctrv':>8s}")
    for p in sorted(per_pattern):
        d = per_pattern[p]
        print(f"{p:20s} {d['ssm_mm']:8.3f} {d['cv']:8.3f} {d['ctrv']:8.3f}")

    print("\n--- Branch-pattern diagnostics (is this model actually multimodal?) ---")
    print(f"go samples ({branch_diag['n_go']}): mode usage {branch_diag['mode_usage_go']}")
    print(f"stop samples ({branch_diag['n_stop']}): mode usage {branch_diag['mode_usage_stop']}")
    fd = branch_diag["mean_final_disp"]
    print(f"true mean final displacement    -- go: {fd['true_go']:.3f} m   stop: {fd['true_stop']:.3f} m")
    print(f"mode0 predicted final displacement -- go: {fd['mode0_on_go']:.3f} m   stop: {fd['mode0_on_stop']:.3f} m")
    print(f"mode1 predicted final displacement -- go: {fd['mode1_on_go']:.3f} m   stop: {fd['mode1_on_stop']:.3f} m")
    print(f"mean top-mode probability -- branch (ambiguous): "
          f"{branch_diag['mean_top_mode_prob_branch']:.3f}   "
          f"constant_velocity (unambiguous): {branch_diag['mean_top_mode_prob_unambiguous']:.3f}")

    os.makedirs("../results", exist_ok=True)
    torch.save(model.state_dict(), "../results/ssm_predictor_multimodal.pt")
    with open("../results/ssm_predictor_multimodal_metrics.json", "w") as f:
        json.dump({"summary": summary, "per_pattern": per_pattern, "branch_diagnostics": branch_diag,
                    "K": K, "H": H, "dt": DT, "n_modes": N_MODES}, f, indent=2)
    print("\nSaved ../results/ssm_predictor_multimodal.pt and "
          "../results/ssm_predictor_multimodal_metrics.json")
