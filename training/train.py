"""
================================================================================
BRICK Training Script
================================================================================

Description:
    Trains a single shared BRICK model on all subjects and sessions
    (pre and post sonication, VIM and ZI targets) simultaneously.
        - One shared BRICK instance (shared K, per-subject-per-session C)
        - Train on all 38 sessions (19 subjects × 2 targets × pre+post)
        - Extract C_pre and C_post after training to compute ΔC

    BRICK training protocol (Zhou et al. 2025):
        - Optimizer: AdamW, lr=1e-3, weight_decay=1e-5
        - Scheduler: CosineAnnealingLR over n_epochs
        - Epochs: 1000
        - Split: 7:1:2 (train/val/test) by subject

    Outputs saved to results/training/:
        - best_model.pt      — checkpoint with lowest validation loss
        - final_model.pt     — checkpoint after last epoch
        - training_log.csv   — per-epoch train and val losses
        - split.json         — subject IDs for each split

Usage:
    python training/train.py
"""

import sys
import json
import random
import logging
import csv
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.brick import BRICK
from preprocessing.load_preprocessed_data import load_all
from config import M, N_ROIS, H, T as T_DATA

# ================================================================================
# CONFIG
# ================================================================================
N_EPOCHS     = 1000
LR           = 1e-3
WEIGHT_DECAY = 1e-5
TRAIN_RATIO  = 0.7
VAL_RATIO    = 0.1
SEED         = 42
RESULTS_DIR  = ROOT_DIR / "results" / "training"

# ================================================================================
# LOGGING
# ================================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log_path = RESULTS_DIR / f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(message)s",
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)


# ================================================================================
# DATA SPLIT
# ================================================================================
def split_subjects(subjects, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO, seed=SEED):
    """
    Split subjects into train/val/test by subject ID (not by session).
    All sessions (VIM/ZI × pre/post) for a given subject stay together.

    Returns three lists of subject dicts.
    """
    # Get unique subject IDs
    unique_ids = sorted(set(s["subject_id"] for s in subjects))
    random.seed(seed)
    random.shuffle(unique_ids)

    n = len(unique_ids)
    n_train = int(n * train_ratio)
    n_val   = int(n * val_ratio)

    train_ids = set(unique_ids[:n_train])
    val_ids   = set(unique_ids[n_train:n_train + n_val])
    test_ids  = set(unique_ids[n_train + n_val:])

    train = [s for s in subjects if s["subject_id"] in train_ids]
    val   = [s for s in subjects if s["subject_id"] in val_ids]
    test  = [s for s in subjects if s["subject_id"] in test_ids]

    return train, val, test


# ================================================================================
# FORWARD PASS FOR ONE SUBJECT
# ================================================================================
def subject_loss(model, subject):
    """
    Run pre and post forward passes for one subject and return combined loss.

    Args:
        model   (BRICK): The model
        subject (dict):  Subject dict from load_all()

    Returns:
        torch.Tensor: Combined loss (pre + post)
    """
    x_pre  = torch.tensor(subject["mpre"],  dtype=torch.float32)   # (T, N)
    x_post = torch.tensor(subject["mpost"], dtype=torch.float32)    # (T, N)

    label_pre  = torch.tensor(0)  # 0 = pre-sonication
    label_post = torch.tensor(1)  # 1 = post-sonication

    out_pre  = model(x_pre,  label_pre)
    out_post = model(x_post, label_post)

    return out_pre["losses"]["loss_total"] + out_post["losses"]["loss_total"]


# ================================================================================
# EPOCH
# ================================================================================
def run_epoch(model, subjects, optimizer=None, train=True):
    """
    Run one epoch over a list of subjects.

    Args:
        model     (BRICK): The model
        subjects  (list):  List of subject dicts
        optimizer:         AdamW optimizer (None in eval mode)
        train     (bool):  If True, update weights

    Returns:
        float: Mean loss over all subjects
    """
    model.train(train)
    total_loss = 0.0

    for subject in subjects:
        loss = subject_loss(model, subject)

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()

    return total_loss / len(subjects)


# ================================================================================
# MAIN TRAINING LOOP
# ================================================================================
def train():
    log.info("=" * 60)
    log.info("BRICK Training")
    log.info(f"N_ROIS={N_ROIS}, M={M}, H={H}, T={T_DATA}")
    log.info(f"Epochs={N_EPOCHS}, LR={LR}, WeightDecay={WEIGHT_DECAY}")
    log.info("=" * 60)

    # --- Load data ---
    log.info("Loading preprocessed data...")
    subjects = load_all()
    log.info(f"Loaded {len(subjects)} sessions")

    # --- Split ---
    train_subjects, val_subjects, test_subjects = split_subjects(subjects)
    log.info(f"Split: {len(train_subjects)} train | {len(val_subjects)} val | {len(test_subjects)} test sessions")

    # Save split for reproducibility
    split_info = {
        "train": sorted(set(s["subject_id"] for s in train_subjects)),
        "val":   sorted(set(s["subject_id"] for s in val_subjects)),
        "test":  sorted(set(s["subject_id"] for s in test_subjects)),
        "seed":  SEED,
    }
    with open(RESULTS_DIR / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)
    log.info(f"Split saved to {RESULTS_DIR / 'split.json'}")

    # --- Model ---
    model = BRICK()
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # --- Optimizer and scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=N_EPOCHS, eta_min=1e-6)

    # --- Training log ---
    csv_path = RESULTS_DIR / "training_log.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "lr"])

    # --- Training loop ---
    best_val_loss = float("inf")
    log.info("Starting training...")

    for epoch in range(1, N_EPOCHS + 1):

        train_loss = run_epoch(model, train_subjects, optimizer, train=True)
        scheduler.step()

        with torch.no_grad():
            val_loss = run_epoch(model, val_subjects, optimizer=None, train=False)

        current_lr = scheduler.get_last_lr()[0]

        # Log to file
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_loss:.6f}", f"{current_lr:.2e}"])

        # Log to terminal every 10 epochs
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{N_EPOCHS} | "
                f"train={train_loss:.4f} | "
                f"val={val_loss:.4f} | "
                f"lr={current_lr:.2e}"
            )

        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                "epoch":      epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss":   val_loss,
                "train_loss": train_loss,
            }, RESULTS_DIR / "best_model.pt")
            log.info(f"  → New best val loss: {val_loss:.4f} (saved best_model.pt)")

    # Save final model
    torch.save({
        "epoch":      N_EPOCHS,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss":   val_loss,
        "train_loss": train_loss,
    }, RESULTS_DIR / "final_model.pt")
    log.info(f"Final model saved to {RESULTS_DIR / 'final_model.pt'}")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info("Training complete.")


if __name__ == "__main__":
    train()