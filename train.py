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
model_config, data_config, train_config = (
    config["model"],
    config["data"],
    config["train"],
)

device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps" if torch.backends.mps.is_available() else "cpu"
)
print(f"device: {device}  config: {config_path}")

train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids = (
    split_data(
        train_task=data_config["train_task"],
        test_task=data_config["test_task"],
        split_percentages=tuple(data_config["split_percentages"]),
        batch_size=data_config["batch_size"],
        cross_patients=data_config["cross_patients"],
        sequence_length=data_config.get("sequence_length", 10),
        device=device,
    )
)

# Build model from config
model = build_model(model_config).to(device)
optimizer = torch.optim.Adam(
    model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

# Train model and save the best state
best_state, best_epoch, train_loss_history, val_loss_history = fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=train_config["epochs"],
    mu=mu,
    sigma=sigma,
    loss=train_config["loss"],
    beta=train_config["beta"],
    patience=train_config["patience"],
)

# Load the best state
model.load_state_dict(best_state)

# evaluate on val set to get the predicted sigma distribution
pred_sigma = evaluate(model, val_loader, mu, sigma, device)["std"]

# build checkpoint to save to file with config, model weights, ...
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
name = f"{model_config['type']}_{data_config['train_task']}v{data_config['test_task']}_beta{train_config['beta']}_ep{train_config['epochs']}"
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
