import os
import sys

import matplotlib.pyplot as plt
import torch
import yaml

from dataset import split_data
from engine import fit
from metrics import evaluate
from models import build_model

# Usage: python train.py [configs/your_config.yaml]
config_path = sys.argv[1] if len(sys.argv) > 1 else "configs/gru_generalist.yaml"
config = yaml.safe_load(open(config_path))
m, d, t = config["model"], config["data"], config["train"]

device = torch.device(
    "cuda" if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available()
    else "cpu"
)
print(f"device: {device}  config: {config_path}")

train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids = split_data(
    train_task=d["train_task"],
    test_task=d["test_task"],
    split_percentages=tuple(d["split_percentages"]),
    batch_size=d["batch_size"],
    cross_patients=d["cross_patients"],
    device=device,
)

model = build_model(m).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=10)

best_state, best_epoch, train_loss_history, val_loss_history = fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=t["epochs"],
    mu=mu,
    sigma=sigma,
    loss=t["loss"],
    beta=t["beta"],
    patience=t["patience"],
)

model.load_state_dict(best_state)

# predicted-σ distribution on the val set at the best weights, in normalized space.
# evaluate.py uses it to pick a percentile threshold for the uncertainty experiment.
pred_sigma = evaluate(model, val_loader, mu, sigma, device)["sigma_norm"]

# the whole config travels with the checkpoint, so eval/compare can rebuild the
# exact model without hardcoding any hyperparameters.
checkpoint = {
    "config": config,
    "model_state": best_state,
    "mu": mu,
    "sigma": sigma,
    "train_ids": train_ids,
    "val_ids": val_ids,
    "test_ids": test_ids,
    "best_epoch": best_epoch,
    "pred_sigma": pred_sigma,
}

out_dir = config.get("output_dir", "checkpoints")
os.makedirs(out_dir, exist_ok=True)
name = f"{m['type']}_{d['train_task']}v{d['test_task']}_beta{t['beta']}_ep{t['epochs']}"
checkpoint_path = os.path.join(out_dir, f"{name}.pth")
torch.save(checkpoint, checkpoint_path)
print(f"saved {checkpoint_path}  (best epoch {best_epoch})")

# train/val loss history plot
plt.figure(figsize=(10, 5))
plt.plot(train_loss_history, label="Train Loss")
plt.plot(val_loss_history, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title(f"Train and Val Loss - {name}")
plt.legend()
plt.grid()
plt.savefig(os.path.join(out_dir, f"{name}_loss_history.png"))
plt.show()
