import os
import sys

import numpy as np
import torch
import torch.nn as nn
import tqdm
import yaml

from dataset import split_data
from metrics import evaluate, fd, fd_gain
from models import build_model, get_device

config_path = sys.argv[1]
print(f"Loading {config_path}...")
config = yaml.safe_load(open(config_path))

student_model_config = config["model"]
train_config = config["train"]
distill_config = config.get("distill", {})

# distillation weights (all tunable from the yaml)
lambda_task = distill_config.get("lambda_task", 1.0)
alpha_out = distill_config.get("alpha_out", 1.0)
alpha_feat = distill_config.get("alpha_feat", 1.0)
temperature = distill_config.get("temperature", 1.0)
feat_normalize = distill_config.get("feat_normalize", True)
beta = train_config["beta"]

device = get_device()

# ---- load the frozen teacher -------------------------------------------------
teacher_path = config["teacher"]
print(f"Loading teacher {teacher_path}...")
teacher_ckpt = torch.load(teacher_path, map_location=device, weights_only=False)
teacher_config = teacher_ckpt["config"]
teacher_in = teacher_config["model"]["input_dim"]

# data augmentation and batch size are overloadable, all the rest is fixed.
data_config = dict(teacher_config["data"])
for k in ("neg_augmentation", "time_augmentation", "batch_size"):
    if k in config.get("data", {}):
        data_config[k] = config["data"][k]
# student and teacher must share the same input_dim
assert teacher_in == student_model_config["input_dim"]

teacher = build_model(teacher_config["model"]).to(device)
teacher.load_state_dict(teacher_ckpt["model_state"])
teacher.eval()
for p in teacher.parameters():
    p.requires_grad_(False)
print(
    f"teacher: {teacher_config['model']['type']}  "
    f"seq_len={data_config['sequence_length']}  "
    f"penultimate dim={teacher.fc_mean.in_features}"
)

# ---- rebuild the teacher's exact split (same ids + mu/sigma) ------------------
(
    train_loader,
    val_loader,
    test_loader,
    mu,
    sigma,
    train_ids,
    val_ids,
    test_ids,
) = split_data(
    train_task=data_config["train_task"],
    test_task=data_config["test_task"],
    split_percentages=tuple(data_config["split_percentages"]),
    batch_size=data_config["batch_size"],
    cross_patients=data_config["cross_patients"],
    sequence_length=data_config["sequence_length"],
    device=device,
    time_augmentation=data_config.get("time_augmentation", False),
    neg_augmentation=data_config.get("neg_augmentation", False),
    add_velocity=data_config.get("add_velocity", False),
    add_acceleration=data_config.get("add_acceleration", False),
    ids=(teacher_ckpt["train_ids"], teacher_ckpt["val_ids"], teacher_ckpt["test_ids"]),
)

# ---- build the student -------------------------------------------------------
student = build_model(student_model_config).to(device)
n_params = sum(p.numel() for p in student.parameters() if p.requires_grad)
print(
    f"student: {student_model_config['type']}  |  trainable params: {n_params:,}  "
    f"|  penultimate dim={student.fc_mean.in_features}"
)

# projection maps the student penultimate into the teacher's penultimate space
proj = None
if alpha_feat > 0:
    proj = nn.Linear(student.fc_mean.in_features, teacher.fc_mean.in_features).to(
        device
    )

# PyTorch lets us attach a function to a layer that it runs automatically on every forward pass, passing
# in that layer's inputs; we just log the input in this dict to read afterwards.
penultimate = {}


def teacher_hook(layer, layer_inputs, layer_output):
    penultimate["teacher"] = layer_inputs[0]


def student_hook(layer, layer_inputs, layer_output):
    penultimate["student"] = layer_inputs[0]


teacher.fc_mean.register_forward_hook(teacher_hook)
student.fc_mean.register_forward_hook(student_hook)


def gaussian_kl(mu_t, var_t, mu_s, var_s, eps=1e-6):
    """KL( N(mu_t, var_t) || N(mu_s, var_s) ), averaged over batch and dims.
    Mean-seeking direction: drives the student mean/var toward the teacher's."""
    var_t = var_t + eps
    var_s = var_s + eps
    kl = (
        0.5 * (torch.log(var_s) - torch.log(var_t))
        + (var_t + (mu_t - mu_s) ** 2) / (2 * var_s)
        - 0.5
    )
    return kl.mean()


def feature_loss(feat_s, feat_t):
    """MSE between the projected student penultimate and the (detached) teacher
    penultimate. Optionally standardize each vector first so the term is
    scale-free and comparable across architectures with different feature norms."""
    feat_s = proj(feat_s)
    feat_t = feat_t.detach()
    if feat_normalize:
        feat_s = (feat_s - feat_s.mean(dim=1, keepdim=True)) / (
            feat_s.std(dim=1, keepdim=True) + 1e-6
        )
        feat_t = (feat_t - feat_t.mean(dim=1, keepdim=True)) / (
            feat_t.std(dim=1, keepdim=True) + 1e-6
        )
    return ((feat_s - feat_t) ** 2).mean()


def mean_fdg(result):
    """Mean per-frame FD-gain from an evaluate() result dict."""
    return float(
        np.mean((result["fd_base"] - result["fd_pred"]) / (result["fd_base"] + 1e-6))
    )


# ---- training ----------------------------------------------------------------
params = list(student.parameters())
if proj is not None:
    params += list(proj.parameters())
optimizer = torch.optim.AdamW(
    params, lr=train_config["lr"], weight_decay=train_config["weight_decay"]
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="min", factor=0.5, patience=10
)

# FD is position-space, so keep only the first 6 (position) channels for
# denormalization -- matching engine.fit / metrics.evaluate.
mu_t = torch.tensor(mu, dtype=torch.float32, device=device)[:6]
sigma_t = torch.tensor(sigma, dtype=torch.float32, device=device)[:6]
nll = nn.GaussianNLLLoss()

best_val_fdg = float("-inf")
best_state = None
best_epoch = 0
early_stop_counter = 0
patience = train_config["patience"]

pbar = tqdm.trange(train_config["epochs"])
for epoch in pbar:
    student.train()
    for patient_ids, x, y in tqdm.tqdm(
        train_loader, leave=False, desc=f"epoch {epoch + 1}"
    ):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            mean_t, var_t = teacher(x)
            feat_t = penultimate["teacher"]

        mean_s, var_s = student(x)
        feat_s = penultimate["student"]

        last_x = x[:, -1, :6]
        fd_base = fd(last_x, y, mu_t, sigma_t)
        fd_pred = fd(mean_s, y, mu_t, sigma_t)
        task_loss = nll(mean_s, y, var_s) - beta * fd_gain(fd_base, fd_pred).mean()

        loss = lambda_task * task_loss
        if alpha_out > 0:
            # temperature softens the teacher's variance (T>1 = flatter target)
            loss = loss + alpha_out * gaussian_kl(
                mean_t, var_t * temperature**2, mean_s, var_s
            )
        if alpha_feat > 0:
            loss = loss + alpha_feat * feature_loss(feat_s, feat_t)

        loss.backward()
        optimizer.step()

    # ---- validation: select on FD-gain, step scheduler on task loss ----------
    val_result = evaluate(student, val_loader, mu, sigma, device)
    val_fdg = mean_fdg(val_result)
    val_loss = val_result["nll"] - beta * val_fdg

    scheduler.step(val_loss)
    pbar.set_postfix({"val_loss": f"{val_loss:.4f}", "val_fdg": f"{val_fdg:.4f}"})

    if val_fdg > best_val_fdg:
        best_val_fdg = val_fdg
        best_epoch = epoch + 1
        early_stop_counter = 0
        best_state = {k: v.detach().clone() for k, v in student.state_dict().items()}
    else:
        early_stop_counter += 1
        if early_stop_counter >= patience:
            break

student.load_state_dict(best_state)

pred_sigma = evaluate(student, val_loader, mu, sigma, device)["std"]

teacher_fdg = mean_fdg(evaluate(teacher, test_loader, mu, sigma, device))
student_fdg = mean_fdg(evaluate(student, test_loader, mu, sigma, device))
print(
    f"\ntest FD-gain  |  teacher {teacher_fdg:.4f}  student {student_fdg:.4f}  "
    f"(best val FD-gain {best_val_fdg:.4f}, epoch {best_epoch})"
)


out_config = {
    "model": student_model_config,
    "data": data_config,
    "train": train_config,
    "teacher": teacher_path,
    "distill": distill_config,
}
checkpoint = {
    "config": out_config,
    "model_state": best_state,
    "mu": mu,
    "sigma": sigma,
    "train_ids": train_ids,
    "val_ids": val_ids,
    "test_ids": test_ids,
    "best_epoch": best_epoch,
    "pred_sigma": pred_sigma,
    "proj_state": proj.state_dict() if proj is not None else None,
}

out_dir = config.get("output_dir", "checkpoints/distill")
os.makedirs(out_dir, exist_ok=True)
name = (
    f"{student_model_config['type']}_distill_from_{teacher_config['model']['type']}"
    f"_{data_config['train_task']}v{data_config['test_task']}_beta{beta}"
    f"_ep{train_config['epochs']}"
)
checkpoint_path = os.path.join(out_dir, f"{name}.pth")
i = 2
while os.path.exists(checkpoint_path):
    checkpoint_path = os.path.join(out_dir, f"{name}_{i}.pth")
    i += 1
torch.save(checkpoint, checkpoint_path)
print(f"saved {checkpoint_path}  (best epoch {best_epoch})")
