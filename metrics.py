import numpy as np
import torch


def fd(pred_frame, true_frame, mu=0, sigma=1):
    # denormalize before computing FD. frame shape: [batch_size, D]
    pred_frame = pred_frame * sigma + mu
    true_frame = true_frame * sigma + mu
    translation_error = torch.abs(pred_frame[:, :3] - true_frame[:, :3]).sum(dim=1)
    # 50 mm (avg head radius) converts radians to mm
    rotation_error = 50 * torch.abs(pred_frame[:, 3:] - true_frame[:, 3:]).sum(dim=1)
    return translation_error + rotation_error


def fd_gain(fd_baseline, fd_pred):
    # fd_baseline and fd_pred shape: [batch_size]
    gain = (fd_baseline - fd_pred) / (
        fd_baseline + 1e-6
    )  # add 1e-6 for numerical stability
    return gain


def evaluate(model, loader, mu, sigma, device, sigma_threshold=None, noise=None):
    model.eval()
    mu = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma = torch.tensor(sigma, dtype=torch.float32, device=device)
    if sigma_threshold is not None:
        sigma_threshold = torch.tensor(
            sigma_threshold, dtype=torch.float32, device=device
        )
    criterion = torch.nn.GaussianNLLLoss()

    # arrays to store (for each sample in the loader): model prediction, target, baseline, std, fd_pred, fd_base, z-score
    #fmt:off
    preds, trues, bases, stds, fd_preds, fd_bases, zs, = [], [], [], [], [], [], []
    #fmt:on
    # ids stores the patient id for each sample, to allow patient aggregation
    ids = []
    # n stores the number of samples
    nll_sum, n = 0.0, 0
    with torch.no_grad():
        for p, x, y in loader:
            x, y = x.to(device), y.to(device)
            if noise is not None:
                x = x + noise * torch.randn_like(x)
            # model returns (mean, variance)
            mean, var = model(x)
            last_x = x[:, -1, :6]  # baseline = previous frame, 6 positions only
            std = var.sqrt() * sigma  # predicted std in physical (denormalized) units

            if sigma_threshold is not None:
                # if the model is uncertain (std above threshold in any dim), use the baseline
                uncertain = (std > sigma_threshold).any(dim=1)
                mean[uncertain] = last_x[uncertain]

            bs = y.size(0)
            nll_sum += criterion(mean, y, var).item() * bs
            n += bs

            fd_preds.append(fd(mean, y, mu, sigma).cpu().numpy())
            fd_bases.append(fd(last_x, y, mu, sigma).cpu().numpy())
            zs.append(((y - mean) / var.sqrt()).cpu().numpy())

            preds.append((mean * sigma + mu).cpu().numpy())
            trues.append((y * sigma + mu).cpu().numpy())
            bases.append((last_x * sigma + mu).cpu().numpy())
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
