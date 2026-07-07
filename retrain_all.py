"""Retrain every checkpoint in a folder from its own embedded config.

Each checkpoint embeds the full config that produced it (`ckpt["config"]`), so
retraining needs no external recipe: we read that config, override epochs and
patience (and redirect output to --out), dump it to a temp yaml, and drive
train.py -- the single training entry point -- so nothing about the recipe is
re-guessed here. The originals are left untouched; fresh checkpoints land in --out.

Unlike retrain_broken.py (which retrains only the miscalibrated ones), this
retrains the whole folder unconditionally -- e.g. to regenerate every checkpoint
after a change to the data/normalization pipeline.

Usage:
  python retrain_all.py checkpoints/generalist
  python retrain_all.py checkpoints/generalist --out checkpoints/generalist_v2
  python retrain_all.py checkpoints/generalist --epochs 150 --patience 30
  python retrain_all.py checkpoints/generalist --dry-run   # list configs, no training
"""

import argparse
import os
import subprocess
import sys
import tempfile

import torch
import yaml

# epochs/patience the retrained runs use, overriding whatever the originals had
EPOCHS = 150
PATIENCE = 30


def retrain(config, out_dir, epochs, patience):
    """Dump the embedded config to a temp yaml (epochs/patience/output_dir
    overridden) and run it through train.py. Returns the checkpoint train.py wrote."""
    run_config = dict(config)
    run_config["output_dir"] = out_dir
    # train.py reads epochs/patience from the train sub-config; copy it before
    # mutating so we never touch the dict loaded from the original checkpoint
    run_config["train"] = dict(run_config["train"])
    run_config["train"]["epochs"] = epochs
    run_config["train"]["patience"] = patience
    os.makedirs(out_dir, exist_ok=True)

    before = set(os.listdir(out_dir))
    with tempfile.NamedTemporaryFile(
        "w", suffix=".yaml", delete=False, dir=out_dir
    ) as f:
        yaml.safe_dump(run_config, f, default_flow_style=False, sort_keys=False)
        yaml_path = f.name
    try:
        # reuse the single training driver; stream its output (tqdm, "saved ...")
        subprocess.run([sys.executable, "train.py", yaml_path], check=True)
    finally:
        os.remove(yaml_path)

    new = [f for f in os.listdir(out_dir) if f.endswith(".pth") and f not in before]
    return os.path.join(out_dir, new[0]) if new else None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("ckpt_dir")
    ap.add_argument(
        "--out", default=None, help="output dir (default: <ckpt_dir>_retrained)"
    )
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--patience", type=int, default=PATIENCE)
    ap.add_argument(
        "--dry-run", action="store_true", help="list configs, don't train"
    )
    args = ap.parse_args()
    out_dir = args.out or args.ckpt_dir.rstrip("/") + "_retrained"

    files = sorted(f for f in os.listdir(args.ckpt_dir) if f.endswith(".pth"))
    if not files:
        print(f"no .pth checkpoints in {args.ckpt_dir}")
        return
    print(
        f"found {len(files)} checkpoint(s) in {args.ckpt_dir} -> retrain with "
        f"epochs={args.epochs} patience={args.patience} -> {out_dir}\n"
    )

    # read each embedded config up front (also surfaces unreadable checkpoints early)
    runs = []  # (fname, config)
    for fname in files:
        ckpt = torch.load(
            os.path.join(args.ckpt_dir, fname), map_location="cpu", weights_only=False
        )
        config = ckpt.get("config")
        if config is None:
            print(f"  ! {fname}: no embedded config, skipping")
            continue
        m, d, t = config["model"], config["data"], config["train"]
        print(
            f"  {fname}\n    type={m['type']} seq_len={d['sequence_length']} "
            f"input_dim={m['input_dim']} loss={t['loss']} beta={t['beta']} lr={t['lr']}"
        )
        runs.append((fname, config))

    if args.dry_run:
        print("\n--dry-run: not training.")
        return

    results = []  # (old_fname, new_fname or None)
    for i, (fname, config) in enumerate(runs, 1):
        print(f"\n{'=' * 70}\n[{i}/{len(runs)}] retraining from {fname}\n{'=' * 70}")
        new_path = retrain(config, out_dir, args.epochs, args.patience)
        if new_path is None:
            print(f"  ! train.py produced no new checkpoint for {fname}")
        else:
            print(f"  -> {os.path.basename(new_path)}")
        results.append((fname, os.path.basename(new_path) if new_path else None))

    print(f"\n{'=' * 70}\nsummary\n{'=' * 70}")
    for old, new in results:
        print(f"  {old:52s} -> {new if new else 'FAILED'}")


if __name__ == "__main__":
    main()
