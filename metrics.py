import numpy as np
import torch


def fd(pred_frame, true_frame, mu=0, sigma=1, scale_rot=True, dimension_dim=1):
    # denormalize before computing FD. frame shape: [batch_size, D]
    pred_frame = pred_frame * sigma + mu
    true_frame = true_frame * sigma + mu
    translation_error = torch.abs(pred_frame[:, :3] - true_frame[:, :3]).sum(
        dim=dimension_dim
    )
    factor = 50 if scale_rot else 1
    rotation_error = factor * torch.abs(pred_frame[:, 3:] - true_frame[:, 3:]).sum(
        dim=dimension_dim
    )
    return translation_error + rotation_error


def fd_gain(fd_baseline, fd_pred):
    # fd_baseline and fd_pred shape: [batch_size]
    gain = (fd_baseline - fd_pred) / (fd_baseline + 1e-6)
    return gain


def evaluate(model, loader, mu, sigma, device, sigma_threshold=None):
    """Run the model over a loader and return per-sample arrays + the mean NLL.

    Returns a dict of numpy arrays, all with batch dimension N:
      pred, true, base, std   [N, 6] in physical (denormalized) units, NO ×50 rotation scaling
      fd_pred, fd_base        [N]    framewise displacement in mm (rotations ×50 inside fd)
      z                       [N, 6] standardized residual (y − μ_pred) / σ_pred, in normalized space
      ids                     [N]    patient id per sample
      nll                     scalar mean Gaussian NLL

    If sigma_threshold (array of 6, physical units) is given, predictions whose
    std exceeds the threshold in any dimension are replaced by the previous-frame
    baseline.
    """
    model.eval()
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    if sigma_threshold is not None:
        sigma_threshold = torch.tensor(
            sigma_threshold, dtype=torch.float32, device=device
        )
    criterion = torch.nn.GaussianNLLLoss()

    preds, trues, bases, stds = [], [], [], []
    fd_preds, fd_bases, zs, ids = [], [], [], []
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for p, x, y in loader:
            x, y = x.to(device), y.to(device)
            # model returns (mean, variance) — variance already exp'd inside the model
            mean, var = model(x)
            last_x = x[:, -1, :]
            std = var.sqrt() * sigma_t  # predicted std in physical (denormalized) units

            if sigma_threshold is not None:
                # if the model is uncertain (std above threshold in any dim), fall back to the baseline
                uncertain = (std > sigma_threshold).any(dim=1)
                mean[uncertain] = last_x[uncertain]

            bs = y.size(0)
            nll_sum += criterion(mean, y, var).item() * bs
            n += bs

            fd_preds.append(fd(mean, y, mu_t, sigma_t).cpu().numpy())
            fd_bases.append(fd(last_x, y, mu_t, sigma_t).cpu().numpy())
            zs.append(((y - mean) / var.sqrt()).cpu().numpy())

            preds.append((mean * sigma_t + mu_t).cpu().numpy())
            trues.append((y * sigma_t + mu_t).cpu().numpy())
            bases.append((last_x * sigma_t + mu_t).cpu().numpy())
            stds.append(std.cpu().numpy())
            ids.extend(list(p))

    return {
        "pred": np.concatenate(preds),
        "true": np.concatenate(trues),
        "base": np.concatenate(bases),
        "std": np.concatenate(stds),
        "fd_pred": np.concatenate(fd_preds),
        "fd_base": np.concatenate(fd_bases),
        "z": np.concatenate(zs),
        "ids": np.array(ids),
        "nll": nll_sum / n,
    }
