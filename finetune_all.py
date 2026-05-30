import csv
import os

import torch
import tqdm

from finetuning import DIMS, finetune_patient, load_task_dicts

"""
=======================================================================================
Fine-tune the pretrained GRU on every held-out patient (one task) and dump a
summary CSV of pretrained-vs-fine-tuned metrics. Plotting lives in finetune_viz.py
so the (expensive) sweep can be run once and re-plotted cheaply.
=======================================================================================
"""

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Fine-tune + evaluate every held-out patient on this one task.
    TASK = "M"
    PRETRAINED_PATH = "checkpoints/generalist/GRU_R+M+LvR+M+L_beta0.5_ep150.pth"

    results_dir = os.path.join("results", "finetune")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, f"ft_{TASK}_before_after.csv")

    pretrained = torch.load(PRETRAINED_PATH, map_location=device, weights_only=False)
    task_dicts = load_task_dicts()

    # Held-out patients (never seen during pretraining). IDs are strings.
    test_ids = [str(i) for i in pretrained["test_ids"]]
    task_dict = task_dicts[TASK]

    rows = []
    for patient_id in tqdm.tqdm(test_ids, desc=f"fine-tuning ({TASK})"):
        if patient_id not in task_dict:
            # patient lacks this task's series — skip
            continue
        row, _ = finetune_patient(patient_id, TASK, pretrained, task_dicts, device)
        rows.append(row)

    # column order: identifiers first, then the comparison metrics
    fieldnames = [
        "patient_id",
        "task",
        "n_train",
        "n_val",
        "n_test",
        "best_epoch",
        "fd_base",
        "fdg_before",
        "fdg_after",
        "delta_fdg",
        "fd_pred_before",
        "fd_pred_after",
        "mse_before",
        "mse_after",
        "pct_improved_before",
        "pct_improved_after",
    ]
    for d in DIMS:
        fieldnames += [f"mse_{d}_before", f"mse_{d}_after"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    # quick summary to stdout
    n = len(rows)
    if n:
        deltas = sorted(r["delta_fdg"] for r in rows)
        mean_delta = sum(deltas) / n
        win_rate = 100 * sum(d > 0 for d in deltas) / n
        q1 = deltas[int(0.25 * (n - 1))]
        q3 = deltas[int(0.75 * (n - 1))]
        print(f"\nSaved {n} patients -> {csv_path}")
        print(f"mean ΔFD-gain = {mean_delta:+.4f} | win rate = {win_rate:.1f}%")
        print(f"ΔFD-gain IQR  = [{q1:+.4f}, {q3:+.4f}]")
    else:
        print(f"No patients processed for task {TASK}.")
