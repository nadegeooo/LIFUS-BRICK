"""
================================================================================
LOSO (Leave-One-Subject-Out) Training  —  19-fold cross-validation
================================================================================

For each of the 19 subjects, trains a fresh shared BRICK model with that
subject's data (both VIM and ZI targets, pre AND post sessions -- 4 items)
held out entirely from training and validation. The remaining 18 subjects
supply the train/val split: 2 subjects (4 items each = 8 items) drawn for
validation/checkpoint-selection, the other 16 subjects (64 items) for
training.

Architecture: use_control=True, use_ic=True, BETA=config.BETA (0.05) --
per-instruction, this LOSO run keeps the IC encoder active (unlike the
earlier use_ic=False/BETA=0 ablation setup used for the main analysis).
These are also train.py's own defaults (use_ic=True unless --no-ic is
passed; config.BETA is already 0.05), so no override is needed beyond
being explicit about it here.

Reuses run_epoch(), get_kl_weights(), safe_save(), setup_logging(), and
CSV_COLUMNS from training/train.py directly rather than reimplementing the
per-epoch training loop -- only the data-split logic is new. train.py's
split_dataset() does a fixed 7:1:2 split by subject and has no notion of
"hold out subject X entirely"; LOSO needs a different split for every fold,
so this script builds its own Subset() objects by filtering on
ds[i]["subject_id"] (the same field train.py itself reads when writing
split.json), rather than calling split_dataset().

Checkpoint per fold: best_model.pt, selected purely by lowest val total
loss -- since that's the sole selection criterion here (no cls_preoverfit
checkpoint for LOSO), early stopping patience is tied directly to val
total loss improvement. This differs from train.py's main script, where
early stopping is tied to the cls_preoverfit joint-improvement criterion
instead; that criterion doesn't apply when we're only tracking best-val-
total, so early stopping needed to be re-anchored to the metric we
actually checkpoint on.

Val-subject selection: 2 subjects drawn from the remaining 18 (excluding
the held-out subject) using a single random.Random(LOSO_SPLIT_SEED)
instance whose state advances across folds -- so the whole 19-fold process
is reproducible from one seed, but each fold draws from a different point
in the stream (an identical val pair across all 19 folds is practically
impossible). Actual draws are logged to fold_manifest.csv and to each
fold's own .log file. If by freak chance the same pair is drawn every
single time, main() prints a warning after the full run (per the
instruction to fall back to a genuinely random, logged seed in that case)
-- it does not auto-resample mid-run, since that would break the
single-seed reproducibility for the folds already completed.

Model re-initialization: each fold trains a fresh BRICK model, reseeded
with the same SEED (from train.py) for weight-init reproducibility across
folds -- i.e. the only thing that differs fold-to-fold is which subject is
held out and which 2 are drawn for val, not the initial weights.

Output: results/loso_19_fold/fold_{subject_id}/
    best_model.pt, final_model.pt, loss_history.csv, split.json
Plus: results/loso_19_fold/fold_manifest.csv (held-out / val / train subjects per fold)

Usage:
    python training/loso_train.py
    python training/loso_train.py --epochs 500            # shorter pilot run
    python training/loso_train.py --subjects 1002 1005    # only these folds (testing/resuming)
"""

import sys
import csv
import json
import random
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.brick import BRICK
from training.dataset import BRICKDataset
from training.train import (
    run_epoch, get_kl_weights, safe_save, setup_logging, CSV_COLUMNS,
    N_EPOCHS, LR, SEED, DATA_DIR,
)
from config import PATIENCE
import config

RESULTS_DIR = ROOT_DIR / "results" / "loso_19_fold"
LOSO_SPLIT_SEED = 42     # fixed seed for val-subject draws; state advances across folds (see docstring)
N_VAL_SUBJECTS = 2


def train_one_fold(ds, held_out_subject, val_rng, all_subject_ids, n_epochs, log_manifest_row):
    """Train a single LOSO fold. Returns best_val_loss for that fold."""
    run_name = f"fold_{held_out_subject}"
    results_dir = RESULTS_DIR / run_name
    log = setup_logging(results_dir, run_name)

    # --- Build LOSO split ---
    remaining = [s for s in all_subject_ids if s != held_out_subject]
    val_subjects = sorted(val_rng.sample(remaining, N_VAL_SUBJECTS))
    train_subjects = sorted(s for s in remaining if s not in val_subjects)

    all_indices = list(range(len(ds)))

    def indices_for(subject_set):
        return [i for i in all_indices if ds[i]["subject_id"] in subject_set]

    held_out_indices = indices_for({held_out_subject})
    train_indices = indices_for(set(train_subjects))
    val_indices = indices_for(set(val_subjects))

    train_ds = Subset(ds, train_indices)
    val_ds = Subset(ds, val_indices)

    log.info("=" * 60)
    log.info(f"LOSO fold -- held out subject {held_out_subject}")
    log.info(f"Val subjects: {val_subjects} ({len(val_indices)} items)")
    log.info(f"Train subjects: {len(train_subjects)} ({len(train_indices)} items)")
    log.info(f"Held-out items (excluded from training entirely): {len(held_out_indices)}")
    log.info(f"use_control=True, use_ic=True, BETA={config.BETA}, "
             f"LAMBDA_NOISE={config.LAMBDA_NOISE}, BATCH_SIZE={config.BATCH_SIZE}")
    log.info("=" * 60)

    log_manifest_row(held_out_subject, val_subjects, train_subjects)

    split_info = {
        "held_out": held_out_subject,
        "val": val_subjects,
        "train": train_subjects,
        "split_seed": LOSO_SPLIT_SEED,
    }
    with open(results_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # --- Model / optimizer (fresh per fold, reseeded for reproducible init) ---
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
    model = BRICK(use_control=True, use_ic=True,
                   lambda_noise=config.LAMBDA_NOISE, beta=config.BETA, epsilon=config.EPSILON)
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=config.WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    csv_path = results_dir / "loss_history.csv"
    with open(csv_path, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=CSV_COLUMNS).writeheader()

    best_val_loss = float("inf")
    epochs_no_improve = 0
    val_losses = train_losses = None
    epoch = 0

    for epoch in range(1, n_epochs + 1):
        kl_g0_weight, kl_u_weight = get_kl_weights(epoch)

        train_losses = run_epoch(
            model, train_ds, train=True, optimizer=optimizer,
            kl_g0_weight=kl_g0_weight, kl_u_weight=kl_u_weight,
            apply_free_bits=True, batch_size=config.BATCH_SIZE,
        )
        with torch.no_grad():
            val_losses = run_epoch(
                model, val_ds, train=False, optimizer=None,
                kl_g0_weight=1.0, kl_u_weight=1.0,
                apply_free_bits=False, batch_size=config.BATCH_SIZE,
            )

        scheduler.step(val_losses["loss_total"])
        current_lr = optimizer.param_groups[0]["lr"]

        row = {"epoch": epoch, "kl_g0_weight": f"{kl_g0_weight:.3f}",
               "kl_u_weight": f"{kl_u_weight:.3f}", "lr": f"{current_lr:.2e}"}
        for k, v in train_losses.items():
            row[f"train_{k}"] = f"{v:.6f}"
        for k, v in val_losses.items():
            row[f"val_{k}"] = f"{v:.6f}"
        with open(csv_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=CSV_COLUMNS).writerow(row)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{n_epochs} | train={train_losses['loss_total']:.4f} | "
                f"val={val_losses['loss_total']:.4f} | lr={current_lr:.2e}"
            )

        val_total = val_losses["loss_total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            safe_save({
                "epoch": epoch, "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss_total": val_losses["loss_total"],
                "train_loss_total": train_losses["loss_total"],
                "use_control": True, "use_ic": True,
                "h": model.h, "m": model.m,
                "held_out_subject": held_out_subject,
            }, results_dir / "best_model.pt")
            log.info(f"  -> New best val loss: {val_total:.4f} (saved best_model.pt)")
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= PATIENCE:
            log.info(f"Early stopping at epoch {epoch} -- no improvement for {PATIENCE} epochs.")
            break

    safe_save({
        "epoch": epoch, "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss_total": val_losses["loss_total"],
        "train_loss_total": train_losses["loss_total"],
        "use_control": True, "use_ic": True,
        "h": model.h, "m": model.m,
        "held_out_subject": held_out_subject,
    }, results_dir / "final_model.pt")

    log.info(f"Fold complete. Best val loss: {best_val_loss:.4f}")
    return best_val_loss


def main(n_epochs: int = N_EPOCHS, subjects=None):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    ds = BRICKDataset(DATA_DIR)
    all_indices = list(range(len(ds)))
    all_subject_ids = sorted(set(ds[i]["subject_id"] for i in all_indices))

    fold_subjects = subjects if subjects else all_subject_ids
    missing = set(fold_subjects) - set(all_subject_ids)
    if missing:
        raise ValueError(f"Requested subjects not found in dataset: {missing}")

    manifest_path = RESULTS_DIR / "fold_manifest.csv"
    manifest_rows = []
    seen_val_pairs = []

    def log_manifest_row(held_out, val_subjects, train_subjects):
        manifest_rows.append({
            "held_out_subject": held_out,
            "val_subjects": ";".join(str(s) for s in val_subjects),
            "n_train_subjects": len(train_subjects),
        })
        seen_val_pairs.append(tuple(val_subjects))

    val_rng = random.Random(LOSO_SPLIT_SEED)
    results = {}
    for held_out in fold_subjects:
        best_val_loss = train_one_fold(
            ds, held_out, val_rng, all_subject_ids, n_epochs, log_manifest_row
        )
        results[held_out] = best_val_loss

    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["held_out_subject", "val_subjects", "n_train_subjects"])
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(f"\nSaved fold manifest: {manifest_path}")

    if len(seen_val_pairs) > 1 and len(set(seen_val_pairs)) == 1:
        print(
            "WARNING: identical val-subject pair was drawn for every fold "
            f"({seen_val_pairs[0]}) despite the advancing RNG stream. "
            "Per the LOSO spec: re-run using a randomized (non-fixed) seed "
            "and record whatever seed was actually used, rather than trusting this run."
        )

    print("\nLOSO training complete. Best val loss per fold:")
    for held_out, loss in results.items():
        print(f"  {held_out}: {loss:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LOSO 19-fold BRICK training")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Only run these held-out subject IDs (testing/resuming); default = all 19")
    args = parser.parse_args()
    main(n_epochs=args.epochs, subjects=args.subjects)