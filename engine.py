import torch
import torch.nn as nn
import tqdm

from metrics import fd, fd_gain


def l2sp(model, reference):
    """L2-SP penalty: pull the trainable weights back toward the reference (theta_0)."""
    penalty = torch.zeros((), device=next(model.parameters()).device)
    for name, param in model.named_parameters():
        # only regularize the layers we actually train (frozen ones excluded)
        if not param.requires_grad:
            continue
        penalty = penalty + ((param - reference[name]) ** 2).sum()
    return penalty


def fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs,
    mu,
    sigma,
    loss="gaussian_nll",
    beta=0.1,
    patience=10,
    reference=None,
    lambda_l2sp=0.0,
):
    """One training loop used for both pretraining and per-patient fine-tuning.

    loss is "gaussian_nll" (uses the variance head) or "mse" (uses the mean only).
    Pass a reference state dict + lambda_l2sp > 0 to add the L2-SP penalty
    (fine-tuning); leave them at the defaults to disable it (pretraining).

    Early stopping and model selection use validation FD-gain (not the loss).
    Returns (best_state, best_epoch, train_loss_history, val_loss_history).
    """
    mu_t = torch.tensor(mu, dtype=torch.float32, device=device)
    sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)
    nll = nn.GaussianNLLLoss()
    mse = nn.MSELoss()

    train_loss_history = []
    val_loss_history = []
    best_val_fdg = float("-inf")
    best_state = None
    best_epoch = 0
    early_counter = 0

    pbar = tqdm.trange(epochs)
    for epoch in pbar:
        model.train()
        # accumulate sample-weighted base loss so the per-sample mean is correct
        # even when the last batch is smaller than the rest
        train_base_sum = torch.zeros((), device=device)
        train_n = 0
        train_fd_bases = []
        train_fd_preds = []
        for _, x, y in train_loader:
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            mean, var = model(x)

            base = nll(mean, y, var) if loss == "gaussian_nll" else mse(mean, y)

            last_x = x[:, -1, :]
            fd_base = fd(last_x, y, mu_t, sigma_t)
            fd_pred = fd(mean, y, mu_t, sigma_t)
            gain = fd_gain(fd_base, fd_pred)

            total = base - beta * gain.mean()
            if reference is not None and lambda_l2sp > 0:
                total = total + lambda_l2sp * l2sp(model, reference)
            total.backward()
            optimizer.step()

            bs = y.size(0)
            train_base_sum += base.detach() * bs
            train_n += bs
            train_fd_bases.append(fd_base.detach())
            train_fd_preds.append(fd_pred.detach())

        train_base = (train_base_sum / train_n).item()
        train_fdg = fd_gain(torch.cat(train_fd_bases), torch.cat(train_fd_preds)).mean().item()
        train_loss = train_base - beta * train_fdg

        model.eval()
        with torch.no_grad():
            val_base_sum = torch.zeros((), device=device)
            val_n = 0
            val_fd_bases = []
            val_fd_preds = []
            for _, x, y in val_loader:
                x, y = x.to(device), y.to(device)
                mean, var = model(x)
                bs = y.size(0)
                val_base_sum += (nll(mean, y, var) if loss == "gaussian_nll" else mse(mean, y)) * bs
                val_n += bs
                last_x = x[:, -1, :]
                val_fd_bases.append(fd(last_x, y, mu_t, sigma_t))
                val_fd_preds.append(fd(mean, y, mu_t, sigma_t))

            val_base = (val_base_sum / val_n).item()
            val_fdg = fd_gain(torch.cat(val_fd_bases), torch.cat(val_fd_preds)).mean().item()
            val_loss = val_base - beta * val_fdg

        scheduler.step(val_loss)
        train_loss_history.append(train_loss)
        val_loss_history.append(val_loss)
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_fdg": f"{train_fdg:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_fdg": f"{val_fdg:.4f}",
            }
        )

        if val_fdg > best_val_fdg:
            best_val_fdg = val_fdg
            best_epoch = epoch + 1
            early_counter = 0
            # clone so later epochs don't mutate the stored best weights
            # (state_dict() returns references to the live parameters)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            early_counter += 1
            if early_counter >= patience:
                break

    return best_state, best_epoch, train_loss_history, val_loss_history
