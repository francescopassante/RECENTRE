import torch


def fd(pred_frame, true_frame, mu, sigma):
    # denormalize before computing FD. frame shape: [batch_size, D]
    pred_frame = pred_frame * sigma + mu
    true_frame = true_frame * sigma + mu
    translation_error = torch.abs(pred_frame[:, :3] - true_frame[:, :3]).sum(dim=1)
    rotation_error = 50 * torch.abs(pred_frame[:, 3:] - true_frame[:, 3:]).sum(dim=1)
    return translation_error + rotation_error


def fd_gain(fd_baseline, fd_pred):
    # fd_baseline and fd_pred shape: [batch_size]
    gain = (fd_baseline - fd_pred) / (fd_baseline + 1e-6)
    return gain
