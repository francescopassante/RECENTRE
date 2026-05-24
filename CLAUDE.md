# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Predicts the next head-motion frame from fMRI head-motion time series (HCP dataset, 6-DOF: 3 translations + 3 rotations). A GRU outputs a Gaussian per dimension (mean + variance) trained with `GaussianNLLLoss` plus a framewise-displacement (FD) gain term. Goal is to beat the "previous-frame" baseline on FD while keeping σ calibrated.

## Common commands

```bash
# Build the per-task .npy dicts from raw HCP txt files (one-time, edit data_paths first)
python preprocessing.py

# Train one model — edit constants in the __main__ block (train_task, test_task, beta, epochs)
python train.py

# Evaluate a single checkpoint and write plots to results/ — edit MODEL_TAG at the top
python test.py

# Sweep all checkpoints in checkpoints/, group by β, write comparison plots to results/beta_scan/
python beta_scan.py
```

No tests, lint, or build step. Outputs land in `results/` (test.py) and `results/beta_scan/` (beta_scan.py). Checkpoints saved as `checkpoints/GRU_train{train_task}_test{test_task}_beta{beta}_ep{epochs}.pth`; a second run of the same config is suffixed `(2)` and `beta_scan.py` aggregates across runs.

## Architecture

**Data flow (`preprocessing.py`):** raw HCP txt → discard last 6 derivative columns → deg→rad on rotation cols → filter to patients that have all three tasks (Resting / Memory / Language) → save one `.npy` dict per task. Each dict is `{patient_id: ndarray[T, 6]}` with fixed `T` per task (Resting=1200, Memory=405, Language=316).

**Split convention (`train.py`):** train on one task (default Resting), then split the *other* task's patient population 50/50 into val and test (non-overlapping patient IDs, seeded `rng = 42`). `mu`/`sigma` are computed on train data per-dimension and used to normalize everything; both are stored in the checkpoint so eval can reproduce normalization without rerunning preprocessing. `test_dict` (the held-out patients' raw frames) is also stashed in the checkpoint so `test.py` and `beta_scan.py` are fully self-contained.

**Windowing (`TimeSeriesDataset.py`):** each sample is a length-10 input subsequence stride-2 (so it spans 20 frames) plus the next frame as target. `time_span = sequence_length * 2`. Number of windows per patient is `T - time_span + 1` — there was a previous off-by-one bug here (noted in the file's comment); preserve the `+1`. `GPUBatchLoader` is a drop-in `DataLoader` replacement that keeps the full dataset resident on GPU and builds each batch with one vectorized gather — much faster than per-sample `__getitem__` + collate when samples already live on device. Use it via `use_gpu_loader = True` in `train.py`.

**Model (`GRU.py`):** 2-layer GRU → LayerNorm → ReLU → FC → LayerNorm → ReLU → Dropout → two heads (`fc_mean`, `fc_logvar`). `forward` returns `(x[:, -1, :] + y_mean, y_logvar.exp())` — i.e. the mean head predicts a **residual** added to the last input frame, and variance is returned already exponentiated (do not exp again in callers).

**Loss:** `loss = GaussianNLLLoss(y_pred, y, y_var) − β · fd_gain.mean()`. β controls the trade-off between likelihood calibration (β=0) and point-prediction FD (β large, e.g. 100). Early stopping and checkpoint selection use **val FD-gain** (not val loss).

**Metrics (`metrics.py`):** `fd(pred, true, mu, sigma)` denormalizes, then sums |Δ| over translation dims and `50 * |Δ|` over rotation dims — the 50 mm factor is the average head radius and converts radians to comparable mm. `fd_gain = (fd_base − fd_pred) / fd_base`. The same 50× scaling is reapplied wherever rotations are visualized in mm (`test.py` and `beta_scan.py`).

**Calibration:** standardized residuals `z = (y − μ_pred) / σ_pred` are scale-invariant, so they are computed in normalized space. Per-dim diagnostics in `test.py` viz 06: mean(z)≈0, std(z)≈1, reduced χ² = mean(z²) ≈ 1, |z|≤1 coverage ≈ 68.3%, |z|≤2 coverage ≈ 95.4%.

## Conventions to preserve

- Dimension order is fixed: `[Tx, Ty, Tz, Rx, Ry, Rz]`. Rotation indices are `3:6` and get the ×50 mm scaling for display/FD.
- `GRU.forward` returns **(mean, variance)**, not (mean, logvar). The exp happens inside the model.
- Checkpoints embed `test_ids`, `test_task`, `mu`, `sigma` — eval scripts read these and reload the per-task dict from `datasets/{test_task}_dict.npy` (no re-running preprocessing).
- `beta_scan.py` parses β from filenames with regex `beta([0-9]+(?:\.[0-9]+)?)_`; keep the `beta<value>_` token in any new checkpoint names.
