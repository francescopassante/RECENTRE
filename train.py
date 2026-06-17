import os
import sys

import torch
import yaml

from dataset import split_data
from engine import fit
from metrics import evaluate
from models import build_model, get_device

# Usage: python train.py config.yaml
config_path = sys.argv[1]

print(f"Loading {config_path}...")
config = yaml.safe_load(open(config_path))
model_config, data_config, train_config = (
    config["model"],
    config["data"],
    config["train"],
)

device = get_device()

# Split patient in train/val/test
train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids = (
    split_data(
        train_task=data_config["train_task"],
        test_task=data_config["test_task"],
        split_percentages=tuple(data_config["split_percentages"]),
        batch_size=data_config["batch_size"],
        cross_patients=data_config["cross_patients"],
        sequence_length=data_config["sequence_length"],
        device=device,
        time_augmentation = data_config.get("time_augmentation", False),
        neg_augmentation = data_config.get("neg_augmentation", False)
    )
)

# Build specified model using config parameters
model = build_model(model_config).to(device)
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model: {model_config['type']}  |  trainable params: {n_params:,}")
optimizer = torch.optim.Adam(
    model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

# Train model and save the best state
best_state, best_epoch = fit(
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
    # optimizer/scheduler state so resume.py can warm-restart at the right LR
    "optimizer_state": optimizer.state_dict(),
    "scheduler_state": scheduler.state_dict(),
}

out_dir = config.get("output_dir", "checkpoints")
os.makedirs(out_dir, exist_ok=True)
name = f"{model_config['type']}_{data_config['train_task']}v{data_config['test_task']}_beta{train_config['beta']}_ep{train_config['epochs']}"
checkpoint_path = os.path.join(out_dir, f"{name}.pth")
torch.save(checkpoint, checkpoint_path)
print(f"saved {checkpoint_path}  (best epoch {best_epoch})")
