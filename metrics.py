import torch


def fd(pred_frame, true_frame, mu=0, sigma=1, scale_rot=True, dimension_dim=1):
    # denormalize before computing FD. frame shape: [batch_size, D]
    pred_frame = pred_frame * sigma + mu
    true_frame = true_frame * sigma + mu
    translation_error = torch.abs(pred_frame[:, :3] - true_frame[:, :3]).sum(
        dim=dimension_dim
    )
    factor = 1 if not scale_rot else 50
    rotation_error = factor * torch.abs(pred_frame[:, 3:] - true_frame[:, 3:]).sum(
        dim=dimension_dim
    )
    return translation_error + rotation_error


def fd_gain(fd_baseline, fd_pred):
    # fd_baseline and fd_pred shape: [batch_size]
    gain = (fd_baseline - fd_pred) / (fd_baseline + 1e-6)
    return gain
