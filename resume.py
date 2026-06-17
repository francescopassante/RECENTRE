import os
import sys

import torch

from dataset import split_data
from engine import fit
from metrics import evaluate
from models import build_model, get_device

# Usage: python resume.py checkpoint.pth [extra_epochs]
checkpoint_path = sys.argv[1]
extra_epochs = int(sys.argv[2]) if len(sys.argv) > 2 else None

print(f"Loading {checkpoint_path}...")
ckpt = torch.load(checkpoint_path, weights_only=False)
config = ckpt["config"]
model_config, data_config, train_config = (
    config["model"],
    config["data"],
    config["train"],
)

device = get_device()

# Reuse the exact split saved in the checkpoint (no reliance on the rng seed)
train_loader, val_loader, test_loader, mu, sigma, train_ids, val_ids, test_ids = (
    split_data(
        train_task=data_config["train_task"],
        test_task=data_config["test_task"],
        split_percentages=tuple(data_config["split_percentages"]),
        batch_size=data_config["batch_size"],
        cross_patients=data_config["cross_patients"],
        sequence_length=data_config["sequence_length"],
        device=device,
        ids=(ckpt["train_ids"], ckpt["val_ids"], ckpt["test_ids"]),
    )
)

# Rebuild the model and load the saved weights
model = build_model(model_config).to(device)
model.load_state_dict(ckpt["model_state"])
n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"model: {model_config['type']}  |  trainable params: {n_params:,}")
print(f"resuming from best epoch {ckpt['best_epoch']}")

optimizer = torch.optim.Adam(
    model.parameters(), lr=train_config["lr"], weight_decay=train_config["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

epochs = extra_epochs if extra_epochs is not None else train_config["epochs"]

# Continue training and keep the best state
best_state, best_epoch = fit(
    model,
    train_loader,
    val_loader,
    optimizer,
    scheduler,
    device,
    epochs=epochs,
    mu=mu,
    sigma=sigma,
    loss=train_config["loss"],
    beta=train_config["beta"],
    patience=train_config["patience"],
)

model.load_state_dict(best_state)

# evaluate on val set to get the predicted sigma distribution
pred_sigma = evaluate(model, val_loader, mu, sigma, device)["std"]

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
name = f"{model_config['type']}_{data_config['train_task']}v{data_config['test_task']}_beta{train_config['beta']}_ep{train_config['epochs']}_resumed"
checkpoint_path = os.path.join(out_dir, f"{name}.pth")
torch.save(checkpoint, checkpoint_path)
print(f"saved {checkpoint_path}  (best epoch {best_epoch})")
