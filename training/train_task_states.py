# training/train_task_states.py
"""
================================================================================
Task-State BRICK Training
================================================================================

Trains two independent BRICK models:
    1. Pre-model  — trained on mpre only, 3-class classifier
    2. Post-model — trained on mpost only, 4-class classifier

Class definitions:
    Pre-model (3 classes):
        0: Pre T1 — first treatment pre-scan (naive, no prior sonication)
        1: Pre VIM T2 — second treatment pre-scan, subject had ZI first
        2: Pre ZI T2  — second treatment pre-scan, subject had VIM first

    Post-model (4 classes):
        0: Post VIM T1 — first VIM sonication (VIM_first group)
        1: Post ZI T1  — first ZI sonication (ZI_first group)
        2: Post VIM T2 — second VIM sonication (ZI_first group)
        3: Post ZI T2  — second ZI sonication (VIM_first group)

Split: 2 subjects held out for val (stratified: 1 VIM_first, 1 ZI_first),
       same subjects for both models, no test set.

Outputs saved to results/training/task_states/{pre_model,post_model}/

Usage:
    python training/train_task_states.py
    python training/train_task_states.py --epochs 1000
"""

import sys
import csv
import json
import logging
import argparse
import random
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import Dataset, Subset

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.brick import BRICK
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS
from training.train import (
    safe_save, setup_logging, get_kl_weights,
    run_epoch, CSV_COLUMNS, LR, SEED, N_EPOCHS,
)
import config

RESULTS_BASE = ROOT_DIR / "results" / "training" / "task_states"


# ================================================================================
# LABEL ASSIGNMENT
# ================================================================================

def get_pre_label(group: str, target: str) -> int:
    """3-class label for mpre items."""
    is_t1 = (group == "VIM_first" and target == "vim") or \
            (group == "ZI_first"  and target == "zi")
    if is_t1:
        return 0  # naive
    elif group == "ZI_first" and target == "vim":
        return 1  # Pre VIM T2 — had ZI already
    elif group == "VIM_first" and target == "zi":
        return 2  # Pre ZI T2 — had VIM already
    raise ValueError(f"Unhandled combination: group={group}, target={target}")


def get_post_label(group: str, target: str) -> int:
    """4-class label for mpost items."""
    if group == "VIM_first" and target == "vim":
        return 0  # Post VIM T1
    elif group == "ZI_first" and target == "zi":
        return 1  # Post ZI T1
    elif group == "ZI_first" and target == "vim":
        return 2  # Post VIM T2
    elif group == "VIM_first" and target == "zi":
        return 3  # Post ZI T2
    raise ValueError(f"Unhandled combination: group={group}, target={target}")


# ================================================================================
# DATASET
# ================================================================================

class TaskStateDataset(Dataset):
    """
    Dataset for task-state classification training.
    Filters to either mpre or mpost only, assigns multi-class labels.

    Args:
        condition: 'pre' or 'post'
    """

    def __init__(self, condition: str):
        assert condition in ("pre", "post"), "condition must be 'pre' or 'post'"
        self.condition = condition
        self.items = []

        subjects = load_all()
        for s in subjects:
            x_key  = "mpre" if condition == "pre" else "mpost"
            x      = torch.tensor(s[x_key], dtype=torch.float32)

            # Z-score per ROI
            x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

            group  = s["group"]
            target = s["target"]

            if condition == "pre":
                label = get_pre_label(group, target)
            else:
                label = get_post_label(group, target)

            self.items.append({
                "x":           x,
                "label":       torch.tensor(label, dtype=torch.long),
                "subject_id":  s["subject_id"],
                "group":       group,
                "target":      target,
            })

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        return self.items[idx]


# ================================================================================
# SPLIT
# ================================================================================

def make_split(ds: TaskStateDataset, val_subjects: list) -> tuple:
    """
    Split dataset into train and val subsets.
    val_subjects: list of subject IDs to hold out for validation.
    No test set.
    """
    val_set = set(val_subjects)
    train_idx, val_idx = [], []

    for i, item in enumerate(ds.items):
        if item["subject_id"] in val_set:
            val_idx.append(i)
        else:
            train_idx.append(i)

    return Subset(ds, train_idx), Subset(ds, val_idx)


def select_val_subjects(condition: str, seed: int = SEED) -> list:
    """
    Select 2 val subjects stratified by group:
        1 from VIM_first, 1 from ZI_first.
    Same subjects used for both pre and post models.
    """
    subjects = load_all()

    # Get unique subjects with their group
    seen = {}
    for s in subjects:
        sid = s["subject_id"]
        if sid not in seen:
            seen[sid] = s["group"]

    vim_first = [sid for sid, g in seen.items() if g == "VIM_first"]
    zi_first  = [sid for sid, g in seen.items() if g == "ZI_first"]

    rng = random.Random(seed)
    vim_first_sorted = sorted(vim_first)
    zi_first_sorted  = sorted(zi_first)
    rng.shuffle(vim_first_sorted)
    rng.shuffle(zi_first_sorted)

    val_subjects = [vim_first_sorted[0], zi_first_sorted[0]]
    return val_subjects


# ================================================================================
# TRAINING LOOP
# ================================================================================

def train_task_model(
    condition:   str,
    num_classes: int,
    val_subjects: list,
    n_epochs:    int   = N_EPOCHS,
    beta:        float = 0.2,
    run_name:    str   = None,
    cls_regression_threshold: float = 0.2,
):
    """
    Train one BRICK model for task-state classification.

    Args:
        condition:   'pre' or 'post'
        num_classes: 3 for pre, 4 for post
        val_subjects: list of subject IDs for validation
        n_epochs:    max training epochs
        beta:        classification loss weight
        run_name:    subfolder name under task_states/
        cls_regression_threshold: early stopping threshold
    """
    if run_name is None:
        run_name = f"{condition}_model"

    results_dir = RESULTS_BASE / run_name
    log = setup_logging(results_dir, run_name)

    log.info("=" * 60)
    log.info(f"Task-State BRICK Training -- {condition.upper()} model")
    log.info(f"num_classes={num_classes}, BETA={beta}, n_epochs={n_epochs}")
    log.info(f"Val subjects: {val_subjects}")
    log.info("=" * 60)

    # --- Data ---
    ds = TaskStateDataset(condition)
    train_ds, val_ds = make_split(ds, val_subjects)
    log.info(f"Split: {len(train_ds)} train | {len(val_ds)} val")

    # Log class distribution
    train_labels = [ds.items[i]["label"].item() for i in train_ds.indices]
    val_labels   = [ds.items[i]["label"].item() for i in val_ds.indices]
    log.info(f"Train label dist: { {l: train_labels.count(l) for l in sorted(set(train_labels))} }")
    log.info(f"Val label dist:   { {l: val_labels.count(l)   for l in sorted(set(val_labels))} }")

    # Save split
    split_info = {
        "train":    sorted(set(ds.items[i]["subject_id"] for i in train_ds.indices)),
        "val":      val_subjects,
        "condition": condition,
        "num_classes": num_classes,
        "seed":     SEED,
    }
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # --- Model ---
    model = BRICK(
        use_control=True,
        use_ic=False,
        beta=beta,
        num_classes=num_classes,
        lambda_noise=config.LAMBDA_NOISE,
        epsilon=config.EPSILON,
    )
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # --- Optimizer and scheduler ---
    optimizer = optim.AdamW(
        model.parameters(), lr=LR, weight_decay=config.WEIGHT_DECAY
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=15, min_lr=1e-6
    )

    # --- CSV ---
    csv_path = results_dir / "loss_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    # --- Adapt run_epoch for TaskStateDataset ---
    # run_epoch expects items with "x" and "lifus_condition" keys.
    # Our dataset uses "label" instead of "lifus_condition".
    # Monkey-patch the dataset items temporarily.
    for item in ds.items:
        item["lifus_condition"] = item["label"]

    # --- Training loop ---
    best_val_loss     = float("inf")
    best_val_cls      = float("inf")
    best_train_cls    = float("inf")

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

        # CSV
        row = {
            "epoch":        epoch,
            "kl_g0_weight": f"{kl_g0_weight:.3f}",
            "kl_u_weight":  f"{kl_u_weight:.3f}",
            "lr":           f"{current_lr:.2e}",
        }
        for k, v in train_losses.items():
            row[f"train_{k}"] = f"{v:.6f}"
        for k, v in val_losses.items():
            row[f"val_{k}"] = f"{v:.6f}"

        with open(csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
            writer.writerow(row)

        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"train={train_losses['loss_total']:.4f} "
                f"(recon={train_losses['loss_recon']:.4f}, "
                f"cls={train_losses['loss_cls']:.4f}) | "
                f"val={val_losses['loss_total']:.4f} "
                f"(cls={val_losses['loss_cls']:.4f}) | "
                f"lr={current_lr:.2e}"
            )

        # Save best recon checkpoint
        if val_losses["loss_total"] < best_val_loss:
            best_val_loss = val_losses["loss_total"]
            safe_save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss_total":   val_losses["loss_total"],
                "use_control":      True,
                "use_ic":           False,
                "h":                model.h,
                "m":                model.m,
                "num_classes":      num_classes,
                "condition":        condition,
            }, results_dir / "best_model_recon.pt")

        # Save best cls checkpoint (joint val + train improvement)
        val_cls_improved   = val_losses["loss_cls"]   < best_val_cls
        train_cls_improved = train_losses["loss_cls"] < best_train_cls
        if val_cls_improved and train_cls_improved:
            best_val_cls   = val_losses["loss_cls"]
            best_train_cls = train_losses["loss_cls"]
            safe_save({
                "epoch":            epoch,
                "model_state_dict": model.state_dict(),
                "val_loss_cls":     val_losses["loss_cls"],
                "train_loss_cls":   train_losses["loss_cls"],
                "val_loss_total":   val_losses["loss_total"],
                "use_control":      True,
                "use_ic":           False,
                "h":                model.h,
                "m":                model.m,
                "num_classes":      num_classes,
                "condition":        condition,
            }, results_dir / "best_model_cls.pt")
            log.info(f"  -> New joint-best cls: val={best_val_cls:.4f}, train={best_train_cls:.4f}")

        # Early stopping on cls regression
        if val_losses["loss_cls"] > best_val_cls + cls_regression_threshold:
            log.info(
                f"Stopping at epoch {epoch} — val cls {val_losses['loss_cls']:.4f} "
                f"exceeds best {best_val_cls:.4f} by >{cls_regression_threshold}"
            )
            break

    # Save final
    safe_save({
        "epoch":            epoch,
        "model_state_dict": model.state_dict(),
        "val_loss_total":   val_losses["loss_total"],
        "use_control":      True,
        "use_ic":           False,
        "h":                model.h,
        "m":                model.m,
        "num_classes":      num_classes,
        "condition":        condition,
    }, results_dir / "final_model.pt")

    log.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log.info(f"Best cls: val={best_val_cls:.4f}, train={best_train_cls:.4f}")
    return best_val_loss


# ================================================================================
# MAIN
# ================================================================================

def main(n_epochs: int = N_EPOCHS, cls_regression_threshold: float = 0.2):
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Same val subjects for both models
    val_subjects = select_val_subjects(condition="pre", seed=SEED)
    print(f"Val subjects (shared): {val_subjects}")

    # Pre-model: 3 classes, mpre only
    train_task_model(
        condition="pre",
        num_classes=3,
        val_subjects=val_subjects,
        n_epochs=n_epochs,
        beta=0.2,
        run_name="pre_model",
        cls_regression_threshold=cls_regression_threshold,
    )

    # Post-model: 4 classes, mpost only
    train_task_model(
        condition="post",
        num_classes=4,
        val_subjects=val_subjects,
        n_epochs=n_epochs,
        beta=0.2,
        run_name="post_model",
        cls_regression_threshold=cls_regression_threshold,
    )


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--cls-threshold", type=float, default=0.2)
    args = parser.parse_args()

    main(n_epochs=args.epochs, cls_regression_threshold=args.cls_threshold)