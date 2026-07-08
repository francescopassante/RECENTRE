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
python finetune.py configs/finetune.yaml
python finetune_plots.py

# Per-frame router (stacking) over frozen experts (best mamba/conformer/gru) + previous-frame baseline
python routing.py          # train the realizable MLP router; reports FD_gain + headroom captured
```

Needs `pyyaml` on top of torch/numpy/matplotlib/scipy/tqdm. No tests or build step. Checkpoints are saved as `{output_dir}/{type}_{train_task}v{test_task}_beta{beta}_ep{epochs}.pth`.

## Layout

A flat set of modules, deliberately simple (no packages, no type hints, no abstraction layers):

- `models.py` — model classes + `MODELS` dict + `build_model(model_config)`. Add an architecture by writing the class and adding one line to `MODELS`.
- `dataset.py` — `TimeSeriesDataset`, `GPUBatchLoader`, `MultiTaskLoader`, `split_data`, `parse_task`.
- `metrics.py` — `fd`, `fd_gain`, and `evaluate()` (the one eval used by every script; returns per-sample arrays + mean NLL).
- `engine.py` — `fit()`, the one training loop, shared by pretraining and per-patient fine-tuning.
- `train.py` / `finetune.py` — drivers, each reads a YAML config.
- `plots.py` (eval figures) / `finetune_plots.py` (CSV figures) / `evaluate.py` / `compare.py`.
- `sweep.py` — shared `run_eval` + `make_figures` engine; `compare.py` (compare a folder of checkpoints grouped by a config field) and `robustness.py` (evaluate under added test-set noise) both drive it.
- `resume.py` — warm-restart training from a checkpoint's optimizer/scheduler state.
- `routing.py` — the per-frame routing (stacking) experiment (see Routing below).
- `analyze_checkpoints.py` — ad-hoc script: print a comparison table over `checkpoints/generalist/`.
- `configs/*.yaml` — the surface you edit: model type + hyperparameters, tasks, loss, β, epochs, finetune knobs.

## Architecture

**Data flow (`preprocessing.py`):** raw HCP txt → discard last 6 derivative columns → deg→rad on rotation cols → filter to patients that have all three tasks (Resting / Memory / Language) → save one `.npy` dict per task. Each dict is `{patient_id: ndarray[T, 6]}` with fixed `T` per task (Resting=1200, Memory=405, Language=316).

**Config-driven checkpoints:** `train.py` embeds the whole config dict in the checkpoint (`ckpt["config"]`), alongside `model_state`, `mu`, `sigma`, `train/val/test_ids`, `best_epoch`, and `pred_sigma` (the val predicted-std distribution in physical units, for the uncertainty experiment). `evaluate.py`/`compare.py`/`finetune.py` rebuild the model with `build_model(ckpt["config"]["model"])` — **no model hyperparameters are hardcoded anywhere in the eval scripts**.

**Split convention (`dataset.py:split_data`):** if train and test tasks overlap, patients are split into disjoint train/val/test sets (seeded `rng=42`) to avoid leakage; otherwise train uses all patients and only val/test are split. `mu`/`sigma` are computed on the training set per feature channel (positions plus any appended velocity/acceleration channels), pooled across tasks, and stored in the checkpoint.

**Normalization (`dataset.py:build_features` + `TimeSeriesDataset`):** there is a single normalization mechanism for every channel. `build_features` appends step-2 velocity (`x[t]−x[t-2]`) and second-difference acceleration channels to the raw positions (zero-padded to length T); then all channels — positions, velocity, acceleration — get the same z-score `(x − mu) / sigma` using one per-channel `mu`/`sigma` (length 6/12/18, matching `input_dim`). There is **no separate per-task velocity/acceleration scale** (the old `feat_std` is gone). Datasets receive raw positions and normalize internally; `split_data` computes `mu`/`sigma` once on the train set and passes them to the val/test datasets (and the eval scripts pass `ckpt["mu"]`/`ckpt["sigma"]`). Because targets/predictions/FD are position-space, `metrics.evaluate` and `engine.fit` slice `mu[:6]`/`sigma[:6]` when denormalizing. This is a breaking change: checkpoints trained before it (which stored `feat_std` and a length-6 `mu`) must be retrained.

**Windowing (`dataset.py`):** each sample is a length-10 input subsequence stride-2 (spanning 20 frames) plus the next frame as target. `time_span = sequence_length * 2`; windows per patient = `T - time_span + 1` (preserve the `+1` — a previous off-by-one was fixed here). `GPUBatchLoader` keeps the dataset on GPU and builds each batch with one vectorized gather; `split_data` uses it via `use_gpu_loader = True`.

**Model (`models.py`):** GRU is 2-layer GRU → LayerNorm → ReLU → FC → LayerNorm → ReLU → Dropout → two heads (`fc_mean`, `fc_logvar`). `forward` returns `(x[:, -1, :] + y_mean, y_logvar.exp())` — the mean head predicts a **residual** added to the last input frame, and variance is returned already exponentiated (do not exp again in callers).

**Loss / training (`engine.py:fit`):** `loss = base − β · fd_gain.mean() (+ λ · L2-SP)`. `base` is GaussianNLL (`loss: gaussian_nll`) or MSE (`loss: mse`, used by fine-tuning). The L2-SP term is added only when `reference` and `lambda_l2sp` are passed (fine-tuning); it vanishes for pretraining. Early stopping and model selection use **val FD-gain**, not the loss.

**Metrics (`metrics.py`):** `fd(pred, true, mu, sigma)` denormalizes, then sums |Δ| over translation dims and `50 * |Δ|` over rotation dims — the 50 mm factor is the average head radius and converts radians to comparable mm. `fd_gain = (fd_base − fd_pred) / (fd_base + 1e-6)`. `evaluate()` returns physical-unit arrays **without** the ×50 rotation scaling; the plotting code applies ×50 for display.

**Calibration:** standardized residuals `z = (y − μ_pred) / σ_pred` are scale-invariant, computed in normalized space. Per-dim diagnostics in `plots.sigma_calibration`: mean(z)≈0, std(z)≈1, reduced χ² = mean(z²) ≈ 1, |z|≤1 ≈ 68.3%, |z|≤2 ≈ 95.4%.

**Routing (`routing.py`):** an experiment in *combining* trained models instead of picking one. This is **stacking / a learned ensemble, not a mixture of experts** — the experts are trained independently and frozen, and only the router on top is trained (a true MoE trains experts and gate jointly so the experts specialize to the routing; nothing here does that). `EXPERTS` (a dict in `routing.py`) names the frozen checkpoints — currently the best mamba/conformer/gru by mean FD_gain, loaded from `checkpoints/generalist/`; the previous-frame baseline is appended as an extra routable option. A small MLP (`Router`) reads per-frame features (each expert's residual `pred − base` and its predicted σ), emits softmax weights over the options, and the blended prediction is trained end-to-end to **minimize FD directly** (experts stay frozen). The script reports each policy's FD-gain against the baseline (single experts, a fixed average, and the trained router) plus the router's mean weight per option. Discipline: the router trains on `val_ids` and is evaluated on `test_ids` (both unseen by every expert), with 15% of val held out for early stopping. **All experts must share the identical seeded R+M+L split and window length** — otherwise the router trains/evals on frames some expert already saw (the split) or the frame sets don't align (the window length); the script asserts the latter. Finding so far: the router beats the best single model and a static ensemble only marginally; most of the per-frame headroom is irreducible noise, so the single best architecture or a fixed average remains the practical choice.

## Conventions to preserve

- Dimension order is fixed: `[Tx, Ty, Tz, Rx, Ry, Rz]`. Rotation indices are `3:6` and get the ×50 mm scaling for display/FD.
- Models return **(mean, variance)**, not (mean, logvar). The exp happens inside the model.
- Checkpoints are self-describing: they embed `config`, `mu`, `sigma`, `test_ids`. Eval reloads the per-task dict from `datasets/{task}_dict.npy` (no re-running preprocessing) and rebuilds the model from the embedded config.
- `evaluate()` is the single evaluation path. Don't reintroduce per-script eval loops.
- When a run produces a new **fundamental** result (a new architecture beating the baseline, a qualitatively new finding, a changed headline number), update `README.md`: copy the relevant figure(s) into `assets/` and add or revise the explanation there. Don't churn the README for routine re-runs — only when the story changes.
