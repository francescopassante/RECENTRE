import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Rotations are scaled by 50 mm (avg head radius) so every dim ends up in mm.
DIM_NAMES = ["Tx (mm)", "Ty (mm)", "Tz (mm)", "Rx (mm)", "Ry (mm)", "Rz (mm)"]
# one marker per dimension, so per-dim points stay distinguishable without inline labels
DIM_MARKERS = ["o", "s", "^", "D", "v", "P"]
# fixed color per task, reused by every task-aware plot below
TASK_COLORS = {"R": "tab:blue", "M": "tab:orange", "L": "tab:green"}


# 1) Per-dimension error: scatter of baseline vs model error (MAE and RMSE)
def error_per_dimension(pred, true, base, frame_task_labels, test_tasks):
    # dimension identity is encoded by marker shape and task by color, so the two
    # legends below replace the inline dim labels (which overlapped near the origin).
    task_handles = [
        Line2D([0], [0], marker="o", linestyle="", color=TASK_COLORS[t], label=t)
        for t in test_tasks
    ] + [Line2D([0], [0], linestyle="--", color="black", label="y = x")]
    dim_handles = [
        Line2D([0], [0], marker=DIM_MARKERS[d], linestyle="", color="gray", label=name)
        for d, name in enumerate(DIM_NAMES)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, label in zip(axes, ["MAE", "RMSE"]):
        hi = 0
        for task in test_tasks:
            m = frame_task_labels == task
            ep = pred[m] - true[m]
            eb = base[m] - true[m]
            if label == "MAE":
                metric_base = np.abs(eb).mean(axis=0)
                metric_model = np.abs(ep).mean(axis=0)
            else:
                metric_base = np.sqrt((eb**2).mean(axis=0))
                metric_model = np.sqrt((ep**2).mean(axis=0))
            for d in range(len(DIM_NAMES)):
                ax.scatter(
                    metric_base[d],
                    metric_model[d],
                    color=TASK_COLORS[task],
                    marker=DIM_MARKERS[d],
                    s=60,
                )
            hi = max(hi, metric_base.max(), metric_model.max())
        hi *= 1.1
        ax.plot([0, hi], [0, hi], "k--")
        ax.set_xlim(0, hi)
        ax.set_ylim(0, hi)
        ax.set_xlabel(f"Baseline {label} (mm)")
        ax.set_ylabel(f"Model {label} (mm)")
        ax.set_title(f"{label} per dimension")
        task_legend = ax.legend(handles=task_handles, loc="upper left", fontsize=8)
        ax.add_artist(task_legend)
        ax.legend(handles=dim_handles, loc="lower right", fontsize=8)
    fig.tight_layout()
    return fig


# 2) True vs Predicted scatter per dimension
def true_vs_predicted(pred, true, frame_task_labels, test_tasks):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("True vs Predicted per dimension")
    for d, ax in enumerate(axes.flat):
        for task in test_tasks:
            m = frame_task_labels == task
            ax.scatter(
                true[m, d],
                pred[m, d],
                s=2,
                alpha=0.3,
                color=TASK_COLORS[task],
                label=task,
            )
        x = true[:, d]
        y = pred[:, d]
        lo = min(x.min(), y.min())
        hi = max(x.max(), y.max())
        ax.plot([lo, hi], [lo, hi], "k--", label="y = x")
        ss_res = ((x - y) ** 2).sum()
        ss_tot = ((x - x.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title(f"{DIM_NAMES[d]}   R²={r2:.3f}")
        ax.legend(fontsize=8, markerscale=4)
    fig.tight_layout()
    return fig


# 3) Framewise-displacement (FD) distribution — model vs baseline
def fd_distribution(fd_pred, fd_base, sample_task_labels, test_tasks):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    ax = axes[0]
    hi = max(np.percentile(fd_pred, 99.5), np.percentile(fd_base, 99.5))
    bins = np.linspace(0, hi, 80)
    # pooled baseline as a common reference, model FD overlaid per task.
    # density=True so tasks with more samples (e.g. Resting) don't dominate.
    ax.hist(
        fd_base,
        bins=bins,
        color="gray",
        alpha=0.4,
        density=True,
        label=f"Baseline (mean={fd_base.mean():.3f})",
    )
    for task in test_tasks:
        m = sample_task_labels == task
        ax.hist(
            fd_pred[m],
            bins=bins,
            color=TASK_COLORS[task],
            histtype="step",
            linewidth=1.5,
            density=True,
            label=f"Model {task} (mean={fd_pred[m].mean():.3f})",
        )
    ax.set_xlabel("Framewise displacement (mm)")
    ax.set_ylabel("density")
    ax.set_title("FD distribution — model (per task) vs baseline")
    ax.legend()

    ax = axes[1]
    fdg_per_sample = (fd_base - fd_pred) / fd_base
    # clip long tails so the central mass is visible
    gain_lim = np.percentile(np.abs(fdg_per_sample), 99)
    gain_bins = np.linspace(-gain_lim, gain_lim, 80)
    for task in test_tasks:
        m = sample_task_labels == task
        g = np.clip(fdg_per_sample[m], -gain_lim, gain_lim)
        ax.hist(
            g,
            bins=gain_bins,
            color=TASK_COLORS[task],
            histtype="step",
            linewidth=1.5,
            density=True,
            label=f"{task} (mean={fdg_per_sample[m].mean():.3f})",
        )
    ax.axvline(0, color="black")
    pos = (fdg_per_sample > 0).mean() * 100
    ax.set_xlabel("FD gain  (baseline − model) / baseline")
    ax.set_ylabel("density")
    ax.set_title(f"Per-sample FD gain — {pos:.1f}% of samples improved")
    ax.legend()
    fig.tight_layout()
    return fig


# 4) Per-patient FD gain — sorted bar chart + baseline vs model FD scatter
def per_patient_fdg(fd_per_patient_pred, fd_per_patient_base, fdg_per_patient, test_tasks):
    # left: one sorted-bar small multiple per task; right: combined FD scatter.
    n_tasks = len(test_tasks)
    fig, axes = plt.subplots(1, n_tasks + 1, figsize=(6 * (n_tasks + 1), 5))
    if n_tasks == 0:
        return fig

    # shared y-range across the bar small multiples (scatter keeps its own scale)
    all_fdg = np.concatenate([fdg_per_patient[t] for t in test_tasks])
    pad = 0.05 * (all_fdg.max() - all_fdg.min())
    y_lo, y_hi = all_fdg.min() - pad, all_fdg.max() + pad

    for ax, task in zip(axes[:n_tasks], test_tasks):
        fdg = fdg_per_patient[task]
        order = np.argsort(fdg)
        sorted_fdg = fdg[order]
        colors = ["blue" if v >= 0 else "red" for v in sorted_fdg]
        ax.bar(np.arange(len(sorted_fdg)), sorted_fdg, color=colors)
        ax.axhline(0, color="black")
        ax.axhline(fdg.mean(), color="black", linestyle="--", label=f"mean = {fdg.mean():.3f}")
        ax.set_ylim(y_lo, y_hi)
        ax.set_xlabel("Patient (sorted by FD gain)")
        ax.set_ylabel("FD gain")
        ax.set_title(f"{task} — {(fdg > 0).mean() * 100:.1f}% of patients improved")
        ax.legend()

    ax = axes[-1]
    hi = 0
    for task in test_tasks:
        ax.scatter(
            fd_per_patient_base[task],
            fd_per_patient_pred[task],
            color=TASK_COLORS[task],
            s=10,
            alpha=0.5,
            label=task,
        )
        hi = max(hi, fd_per_patient_base[task].max(), fd_per_patient_pred[task].max())
    hi *= 1.05
    ax.plot([0, hi], [0, hi], "k--", label="y = x")
    ax.set_xlim(0, hi)
    ax.set_ylim(0, hi)
    ax.set_xlabel("Baseline FD (per patient)")
    ax.set_ylabel("Model FD (per patient)")
    ax.set_title("Per-patient baseline vs model FD")
    ax.legend()
    fig.tight_layout()
    return fig


# 5) Metrics summary card
def metrics_summary(pred, true, base, task_scalars, test_tasks, tag):
    err_pred = pred - true
    err_base = base - true
    mae_pred_dim = np.abs(err_pred).mean(axis=0)
    mae_base_dim = np.abs(err_base).mean(axis=0)
    rmse_pred_dim = np.sqrt((err_pred**2).mean(axis=0))
    rmse_base_dim = np.sqrt((err_base**2).mean(axis=0))
    total_mae_pred = np.abs(err_pred).mean()
    total_mae_base = np.abs(err_base).mean()
    total_rmse_pred = np.sqrt((err_pred**2).mean())
    total_rmse_base = np.sqrt((err_base**2).mean())

    fig, ax = plt.subplots(figsize=(11, 6))
    ax.axis("off")
    fig.suptitle(f"Test summary — {tag}")

    # headline metrics averaged across tasks (identical to the single-task values
    # when only one task is evaluated)
    overall_nll = np.mean([task_scalars[t]["nll"] for t in test_tasks])
    overall_fd_base = np.mean([task_scalars[t]["fd_base"] for t in test_tasks])
    overall_fd_pred = np.mean([task_scalars[t]["fd_pred"] for t in test_tasks])
    overall_fdg = np.mean([task_scalars[t]["fdg"] for t in test_tasks])

    headline = (
        f"NLL = {overall_nll:.4f}     "
        f"FD baseline = {overall_fd_base:.4f}     "
        f"FD model = {overall_fd_pred:.4f}     "
        f"FD gain = {overall_fdg:.4f}"
    )
    ax.text(0.5, 0.93, headline, ha="center", va="top", fontsize=12, weight="bold")

    # per-task breakdown under the headline
    task_line = "     ".join(
        f"{t}: NLL={task_scalars[t]['nll']:.3f} FDg={task_scalars[t]['fdg']:.3f}"
        for t in test_tasks
    )
    ax.text(0.5, 0.87, task_line, ha="center", va="top", fontsize=9)

    # table of per-dim metrics
    table_data = [["Dim", "MAE base", "MAE model", "RMSE base", "RMSE model"]]
    for d, name in enumerate(DIM_NAMES):
        table_data.append(
            [
                name,
                f"{mae_base_dim[d]:.3f}",
                f"{mae_pred_dim[d]:.3f}",
                f"{rmse_base_dim[d]:.3f}",
                f"{rmse_pred_dim[d]:.3f}",
            ]
        )
    table_data.append(
        [
            "TOTAL",
            f"{total_mae_base:.3f}",
            f"{total_mae_pred:.3f}",
            f"{total_rmse_base:.3f}",
            f"{total_rmse_pred:.3f}",
        ]
    )
    tbl = ax.table(
        cellText=table_data[1:],
        colLabels=table_data[0],
        cellLoc="center",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.78],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    return fig


# 6) Predicted-sigma calibration — standardized residuals z = (y − μ_pred) / σ_pred
def sigma_calibration(z, sample_task_labels, test_tasks):
    # If σ_pred is well-calibrated, z should be ~ N(0, 1) per dimension.
    #   empirical std(z) > 1  ⇒  model is overconfident (σ_pred too small)
    #   empirical std(z) < 1  ⇒  model is underconfident (σ_pred too large)
    #   reduced χ² = mean(z²) should be ≈ 1
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle("Predicted-σ calibration — standardized residuals z = (y − μ_pred) / σ_pred")
    zz = np.linspace(-5, 5, 400)
    gauss_pdf = np.exp(-0.5 * zz**2) / np.sqrt(2 * np.pi)
    for d, ax in enumerate(axes.flat):
        for task in test_tasks:
            m = sample_task_labels == task
            # clip extreme tails for the histogram view
            zd_clip = np.clip(z[m, d], -5, 5)
            ax.hist(
                zd_clip,
                bins=80,
                density=True,
                color=TASK_COLORS[task],
                histtype="step",
                linewidth=1.5,
                label=task,
            )
        ax.plot(zz, gauss_pdf, "k--", label="N(0, 1)")
        # stats pooled across tasks
        zd = z[:, d]
        mean_z = zd.mean()
        std_z = zd.std()
        chi2_red = (zd**2).mean()
        cov68 = (np.abs(zd) <= 1.0).mean() * 100  # should be ~68.3% if calibrated
        cov95 = (np.abs(zd) <= 2.0).mean() * 100  # should be ~95.4%
        ax.set_title(
            f"{DIM_NAMES[d]}   mean={mean_z:.2f}  std={std_z:.2f}  χ²ᵣ={chi2_red:.2f}\n|z|≤1: {cov68:.1f}%   |z|≤2: {cov95:.1f}%",
            fontsize=9,
        )
        ax.set_xlabel("z")
        ax.set_ylabel("density")
        ax.set_xlim(-5, 5)
        if d == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# 7) Random patient — time series with predictive uncertainty band
def patient_timeseries(true_p, pred_p, base_p, std_p, patient_id, viz_task):
    # arrays are this patient's frames in time order, already in mm (rotations ×50)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    fig.suptitle(f"Predicted vs True per dimension — patient {patient_id} ({viz_task})")
    t_axis = np.arange(true_p.shape[0])
    for d, ax in enumerate(axes.flat):
        ax.fill_between(
            t_axis,
            pred_p[:, d] - std_p[:, d],
            pred_p[:, d] + std_p[:, d],
            color="blue",
            alpha=0.3,
            label="±1σ",
        )
        ax.plot(t_axis, true_p[:, d], color="black", label="True", alpha=0.5)
        ax.plot(t_axis, pred_p[:, d], color="blue", label="Predicted", alpha=0.1)
        ax.plot(t_axis, base_p[:, d], color="red", label="Baseline", alpha=0.1)
        ax.set_xlabel("Time step")
        ax.set_ylabel(DIM_NAMES[d])
        ax.set_title(DIM_NAMES[d])
        if d == 0:
            ax.legend(fontsize=8)
    fig.tight_layout()
    return fig


# 8) FD-gain vs motion magnitude — does the model still help on high-motion frames?
def fdgain_vs_motion(fd_pred, fd_base, sample_task_labels, test_tasks):
    # Bin frames by baseline FD (the true previous→next motion). A next-frame
    # predictor can "win" on calm frames yet fail on the large motions that
    # actually corrupt the scan, so we check whether the gain holds as motion grows.
    # Quantile bins give roughly equal counts per bin despite the heavy FD tail.
    n_bins = 10
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    hi = 0
    for task in test_tasks:
        m = sample_task_labels == task
        b = fd_base[m]
        p = fd_pred[m]
        edges = np.unique(np.quantile(b, np.linspace(0, 1, n_bins + 1)))
        idx = np.clip(np.digitize(b, edges[1:-1]), 0, len(edges) - 2)
        centers, model_fd, gain = [], [], []
        for k in range(len(edges) - 1):
            sel = idx == k
            if not sel.any():
                continue
            mean_base = b[sel].mean()
            mean_pred = p[sel].mean()
            centers.append(mean_base)
            model_fd.append(mean_pred)
            # aggregate gain per bin (robust to per-frame fd_base≈0)
            gain.append((mean_base - mean_pred) / mean_base)
        centers = np.array(centers)
        model_fd = np.array(model_fd)
        gain = np.array(gain)

        axes[0].plot(centers, model_fd, "o-", color=TASK_COLORS[task], label=task)
        axes[1].plot(centers, gain, "o-", color=TASK_COLORS[task], label=task)
        hi = max(hi, centers.max(), model_fd.max())

    # left: model FD vs motion magnitude, with the previous-frame baseline (y = x);
    # points below the line mean the model beats the baseline at that motion level.
    hi *= 1.05
    axes[0].plot([0, hi], [0, hi], "k--", label="baseline (y = x)")
    axes[0].set_xlim(0, hi)
    axes[0].set_ylim(0, hi)
    axes[0].set_xlabel("Baseline FD = motion magnitude (mm)")
    axes[0].set_ylabel("Mean model FD (mm)")
    axes[0].set_title("Model FD vs motion magnitude")
    axes[0].legend()

    # right: FD-gain per motion bin — does the gain survive at high motion?
    axes[1].axhline(0, color="black")
    axes[1].set_xlabel("Baseline FD = motion magnitude (mm)")
    axes[1].set_ylabel("FD-gain  (baseline − model) / baseline")
    axes[1].set_title("FD-gain vs motion magnitude")
    axes[1].legend()
    fig.tight_layout()
    return fig


# 10) Predicted-σ vs FD — does high σ flag frames where the baseline beats the model?
# The right panel is the decision view for a "high σ → fall back to baseline" rule:
# wherever the model line rises above the baseline line, those frames are ones the
# fallback would improve.
def fd_vs_sigma(std, fd_pred, fd_base, sample_task_labels, test_tasks):
    # σ per frame on the same mm scale as FD: sum translation σ + 50·sum rotation σ,
    # mirroring how fd() combines the six dims, so "high σ" is directly comparable
    # to a framewise displacement.
    sigma_fd = std[:, :3].sum(axis=1) + 50 * std[:, 3:].sum(axis=1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # left: 2D density of model FD vs σ. A raw scatter of every frame (×3 tasks,
    # plus baseline) is an unreadable hairball, so we hexbin the joint distribution
    # instead — log counts show where the mass sits and how FD spreads at each σ.
    ax = axes[0]
    x_hi = np.percentile(sigma_fd, 99.5)
    y_hi = np.percentile(fd_pred, 99)
    keep = (sigma_fd <= x_hi) & (fd_pred <= y_hi)
    hb = ax.hexbin(
        sigma_fd[keep], fd_pred[keep], gridsize=45, bins="log", cmap="viridis", mincnt=1
    )
    fig.colorbar(hb, ax=ax, label="frame count (log)")
    ax.set_xlabel("Predicted σ (FD-weighted, mm)")
    ax.set_ylabel("Model framewise displacement (mm)")
    ax.set_title("Model FD vs predicted σ — density")

    # right: σ-quantile bins (pooled across tasks). Compare mean model FD vs mean
    # baseline FD per bin: where the model line rises above the baseline line, a
    # high-σ → baseline fallback would improve those frames. Quantile bins give
    # roughly equal counts despite the heavy σ tail.
    n_bins = 12
    edges = np.unique(np.quantile(sigma_fd, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(sigma_fd, edges[1:-1]), 0, len(edges) - 2)
    centers, mean_pred, mean_base = [], [], []
    for k in range(len(edges) - 1):
        sel = idx == k
        if not sel.any():
            continue
        centers.append(sigma_fd[sel].mean())
        mean_pred.append(fd_pred[sel].mean())
        mean_base.append(fd_base[sel].mean())
    centers = np.array(centers)
    mean_pred = np.array(mean_pred)
    mean_base = np.array(mean_base)

    ax = axes[1]
    ax.plot(centers, mean_base, "o-", color="gray", label="Baseline FD")
    ax.plot(centers, mean_pred, "o-", color="tab:red", label="Model FD")
    # shade the σ range where the baseline beats the model (fallback would help)
    ax.fill_between(
        centers, mean_base, mean_pred, where=mean_pred > mean_base,
        color="red", alpha=0.15, label="model worse than baseline",
    )
    y_top = ax.get_ylim()[1]
    for pct in (90, 95):
        xp = np.percentile(sigma_fd, pct)
        ax.axvline(xp, color="black", linestyle=":", linewidth=1)
        ax.text(xp, y_top, f"p{pct}", fontsize=8, va="top", ha="right")
    ax.set_xlabel("Predicted σ (FD-weighted, mm)")
    ax.set_ylabel("Mean FD per σ bin (mm)")
    ax.set_title("Mean FD vs σ — model vs baseline")
    ax.legend(fontsize=8)

    fig.tight_layout()
    return fig


# 9) Model profile — size / parameters / FLOPs / inference time, as a plain table.
# rows is a list of [metric, value] pairs already formatted as strings in evaluate.py.
def model_profile(rows, tag):
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.axis("off")
    fig.suptitle(f"Model profile — {tag}")

    tbl = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        cellLoc="left",
        colLoc="left",
        loc="center",
        bbox=[0.05, 0.05, 0.9, 0.85],
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.auto_set_column_width([0, 1])
    return fig
