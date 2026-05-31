# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this project does

Predicts the next head-motion frame from fMRI head-motion time series (HCP dataset, 6-DOF: 3 translations + 3 rotations). A GRU outputs a Gaussian per dimension (mean + variance) trained with `GaussianNLLLoss` plus a framewise-displacement (FD) gain term. Goal is to beat the "previous-frame" baseline on FD while keeping σ calibrated.

## Common commands

```bash
# Build the per-task .npy dicts from raw HCP txt files (one-time, edit data_paths first)
python preprocessing.py

# Train one model — everything is specified by the YAML config you pass
python train.py configs/gru_generalist.yaml

# Evaluate a checkpoint and write the 8 figures to results/ — config rebuilt from the checkpoint
python evaluate.py checkpoints/generalist/gru_R+M+LvR+M+L_beta0.5_ep150.pth

# Compare a folder of checkpoints, grouped by one config field (default train.beta)
python compare.py checkpoints/beta_scan

# Per-patient fine-tuning sweep -> CSV, then plot it
python finetune.py configs/gru_finetune.yaml
python finetune_plots.py
```

Needs `pyyaml` on top of torch/numpy/matplotlib/scipy/tqdm. No tests or build step. Checkpoints are saved as `{output_dir}/{type}_{train_task}v{test_task}_beta{beta}_ep{epochs}.pth`.

## Layout

A flat set of modules, deliberately simple (no packages, no type hints, no abstraction layers):

- `models.py` — model classes + `MODELS` dict + `build_model(model_config)`. Add an architecture by writing the class and adding one line to `MODELS`.
- `dataset.py` — `TimeSeriesDataset`, `GPUBatchLoader`, `MultiTaskLoader`, `split_data`, `parse_task`.
- `metrics.py` — `fd`, `fd_gain`, and `evaluate()` (the one eval used by every script; returns per-sample arrays + mean NLL).
- `engine.py` — `fit()`, the one training loop, shared by pretraining and per-patient fine-tuning.
- `train.py` / `finetune.py` — drivers, each reads a YAML config.
- `plots.py` (8 eval figures) / `finetune_plots.py` (CSV figures) / `evaluate.py` / `compare.py`.
- `configs/*.yaml` — the surface you edit: model type + hyperparameters, tasks, loss, β, epochs, finetune knobs.

## Architecture

**Data flow (`preprocessing.py`):** raw HCP txt → discard last 6 derivative columns → deg→rad on rotation cols → filter to patients that have all three tasks (Resting / Memory / Language) → save one `.npy` dict per task. Each dict is `{patient_id: ndarray[T, 6]}` with fixed `T` per task (Resting=1200, Memory=405, Language=316).

**Config-driven checkpoints:** `train.py` embeds the whole config dict in the checkpoint (`ckpt["config"]`), alongside `model_state`, `mu`, `sigma`, `train/val/test_ids`, `best_epoch`, and `pred_sigma` (the val predicted-std distribution in physical units, for the uncertainty experiment). `evaluate.py`/`compare.py`/`finetune.py` rebuild the model with `build_model(ckpt["config"]["model"])` — **no model hyperparameters are hardcoded anywhere in the eval scripts**.

**Split convention (`dataset.py:split_data`):** if train and test tasks overlap, patients are split into disjoint train/val/test sets (seeded `rng=42`) to avoid leakage; otherwise train uses all patients and only val/test are split. `mu`/`sigma` are computed on train frames per-dimension and stored in the checkpoint.

**Windowing (`dataset.py`):** each sample is a length-10 input subsequence stride-2 (spanning 20 frames) plus the next frame as target. `time_span = sequence_length * 2`; windows per patient = `T - time_span + 1` (preserve the `+1` — a previous off-by-one was fixed here). `GPUBatchLoader` keeps the dataset on GPU and builds each batch with one vectorized gather; `split_data` uses it via `use_gpu_loader = True`.

**Model (`models.py`):** GRU is 2-layer GRU → LayerNorm → ReLU → FC → LayerNorm → ReLU → Dropout → two heads (`fc_mean`, `fc_logvar`). `forward` returns `(x[:, -1, :] + y_mean, y_logvar.exp())` — the mean head predicts a **residual** added to the last input frame, and variance is returned already exponentiated (do not exp again in callers).

**Loss / training (`engine.py:fit`):** `loss = base − β · fd_gain.mean() (+ λ · L2-SP)`. `base` is GaussianNLL (`loss: gaussian_nll`) or MSE (`loss: mse`, used by fine-tuning). The L2-SP term is added only when `reference` and `lambda_l2sp` are passed (fine-tuning); it vanishes for pretraining. Early stopping and model selection use **val FD-gain**, not the loss.

**Metrics (`metrics.py`):** `fd(pred, true, mu, sigma)` denormalizes, then sums |Δ| over translation dims and `50 * |Δ|` over rotation dims — the 50 mm factor is the average head radius and converts radians to comparable mm. `fd_gain = (fd_base − fd_pred) / (fd_base + 1e-6)`. `evaluate()` returns physical-unit arrays **without** the ×50 rotation scaling; the plotting code applies ×50 for display.

**Calibration:** standardized residuals `z = (y − μ_pred) / σ_pred` are scale-invariant, computed in normalized space. Per-dim diagnostics in `plots.sigma_calibration`: mean(z)≈0, std(z)≈1, reduced χ² = mean(z²) ≈ 1, |z|≤1 ≈ 68.3%, |z|≤2 ≈ 95.4%.

## Conventions to preserve

- Dimension order is fixed: `[Tx, Ty, Tz, Rx, Ry, Rz]`. Rotation indices are `3:6` and get the ×50 mm scaling for display/FD.
- Models return **(mean, variance)**, not (mean, logvar). The exp happens inside the model.
- Checkpoints are self-describing: they embed `config`, `mu`, `sigma`, `test_ids`. Eval reloads the per-task dict from `datasets/{task}_dict.npy` (no re-running preprocessing) and rebuilds the model from the embedded config.
- `evaluate()` is the single evaluation path. Don't reintroduce per-script eval loops.
