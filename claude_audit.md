# Model audit & proposal

Audit of the current architectures in `models.py` against the leaderboard, plus a
proposed new model. Generated from the code + the reported FD_gain table (configs
were intentionally not consulted for the audit itself).

## Leaderboard, grouped by family

| Family | Best FD_gain | Where it tops out | Note |
|---|---|---|---|
| **GRU** | **0.1932** | seqlen 128, +velocity | best by a clear margin |
| Transformer | 0.1696 | seqlen 64, +accel | plateaus ~0.13–0.17 |
| Conformer | 0.1619 | seqlen **10** only, 784k params | never tried long |
| TCN | 0.1598 | seqlen 64 | plateaus |
| TSMixer / PatchTST / DLinear / NLinear | 0.05–0.08 | — | effectively fail |

## Three levers the table exposes

1. **Context length is the dominant factor.** The GRU is monotone in sequence
   length: `10 → 0.145`, `32 → 0.174`, `64 → 0.188`, `128 → 0.193`. Nothing else
   produces a lift of that size. Returns are diminishing, though: `+0.013` from
   32→64 but only `+0.0055` from 64→128 — context alone is near saturation for the
   GRU.
2. **Derivative features help every family.** Velocity (`input_dim=12`) and
   acceleration (`input_dim=18`) variants consistently beat the position-only
   ones. Consistent with the prediction contract (predict the next *delta*).
3. **Inductive bias matters: recurrence > attention > conv > MLP-mixing/linear.**
   The pure channel-mixing / linear models (TSMixer, DLinear, NLinear, PatchTST)
   collapse to 0.05–0.08 — they cannot model the nonlinear temporal dynamics that
   the FD-gain objective rewards.

## Why attention / conv plateau while the GRU wins

Every model uses the same contract: read the **last timestep**, predict a
**residual on the last frame**, output a per-dim variance. That contract is
essentially "estimate the next delta." Recurrence has exactly the right inductive
bias for it — sequential integration of smooth motion — so it gets there with the
least friction. The transformer must rediscover locality and ordering through
positional encodings, and its quadratic cost caps the context that turned out to
matter most. The TCN's fixed dilated receptive field is similar. Neither captured
the long-context advantage that carried the GRU.

## What's underexplored

- **Long context (128) was only ever tried on GRU and Transformer.** The Conformer
  — the most expensive model at 784k params — was only run at seqlen 10.
- **No state-space model exists at all** — the one family purpose-built for the
  thing that is winning here (long sequential context, linear cost).
- The GRU is **unidirectional**, even though the input window is entirely past
  relative to the (out-of-window) target, so there is no causality constraint
  inside the window.

## Proposal: a selective state-space model (Mamba / S6)

The natural next architecture given the audit. It targets the winning signal —
long sequential context — without the GRU's bottleneck.

- **Keeps the inductive bias that wins.** It is recurrent at heart (a linear SSM
  recurrence), so it inherits the GRU's advantage rather than fighting it like
  attention does.
- **Uses the *same* context better, not just more.** The GRU compresses the whole
  window into one fixed hidden vector (`h_n[-1]`). An SSM carries a structured
  diagonal state with input-dependent (*selective*) gating, so it can hold
  long-range information *and* focus on the predictive recent dynamics while
  ignoring slow drift. Since GRU context gains already saturate near 128, "better
  use of the same context" is the right lever.
- **Linear-time**, so long context is cheap (no quadratic attention wall).
- **Fuses the two ingredients that individually helped.** A Mamba block has a
  short depthwise causal conv (local motion spikes, like the Conformer/TCN conv)
  *before* the selective scan (long-range memory).

It drops straight into the repo contract: `input_proj → N blocks → take last
timestep → shared two-head MLP → residual on x[:, -1, :6]`, variance exponentiated.
Head names stay `fc1/bn_fc1/fc_mean/fc_logvar` so fine-tuning's freeze logic works.

Target to beat: **0.1932** (GRU @ seqlen 128, +velocity), evaluated at the same
context length and feature set so the only variable is the backbone.

### Implementation note (pure PyTorch)

The selective scan is implemented as a sequential loop over timesteps
(`mamba-minimal` style): correct but launch/memory-bound at long L. For a
production run, swap the loop for the fused `mamba-ssm` CUDA kernel or a chunked
parallel scan. The materialized `[B, L, d_inner, d_state]` tensors dominate memory
— scale `batch_size` (and optionally `d_state`) to the GPU.

## Secondary, lower-risk option: bidirectional GRU

Extend the proven winner instead of adding a family. Run the GRU in both
directions over the window and concat the two final states into the head. The
window is entirely past relative to the (out-of-window) target, so reading it
backwards leaks nothing — yet the current GRU only reads it forward. The backward
pass gives the head an unforgotten view of the early-window trend (relevant given
long context helped). Likely a small, cheap gain; not guaranteed to beat a
well-tuned unidirectional GRU, since one-step-ahead prediction is dominated by the
most-recent frame, which the forward GRU already places freshest in its state.
