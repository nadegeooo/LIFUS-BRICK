"""
================================================================================
BRICK Training Script
================================================================================

Description:
    Trains a single shared BRICK model on all subjects and sessions
    (pre and post sonication, VIM and ZI targets) simultaneously.
        - One shared BRICK instance (shared K, per-subject-per-session C)
        - Train on all 76 items (19 subjects x 2 targets x pre + post)
        - Extract C_pre and C_post after training to compute Delta_C

    Training protocol (Zhou et al. 2025):
        - Optimizer:  AdamW, lr=1e-3, weight_decay=1e-5
        - Scheduler:  CosineAnnealingLR over n_epochs
        - Early stop: patience=50 epochs on total validation loss
        - Split:      7:1:2 by subject (no session leakage)

    KL Annealing (training stabilization for N=19, not in original BRICK):
        - KL_g0: linear ramp from 0 to 1 over KL_G0_ANNEAL_EPOCHS
        - KL_u:  held at 0 for KL_U_DELAY_EPOCHS, then ramped over KL_U_ANNEAL_EPOCHS
        - Free bits applied during training only (apply_free_bits=True)
        - Validation uses true ELBO (apply_free_bits=False) for honest evaluation

    Outputs saved to results/training/{run_name}/:
        - best_model.pt      — checkpoint with lowest validation loss
        - final_model.pt     — checkpoint after last epoch / early stop
        - loss_history.csv   — per-epoch logging of all loss components
        - split.json         — subject IDs for each split

Usage:
    python training/train.py
    python training/train.py --epochs 100   # pilot run
    python training/train.py --no-control   # ablation: no control module
    python training/train.py --no-ic        # ablation: no IC module
"""

import sys
import csv
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.optim as optim

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.brick import BRICK
from training.dataset import BRICKDataset, split_dataset
from config import (
    M, N_ROIS, H, T as T_DATA,
    KL_G0_ANNEAL_EPOCHS, KL_G0_DELAY_EPOCHS, KL_U_DELAY_EPOCHS, KL_U_ANNEAL_EPOCHS,
    PATIENCE, WEIGHT_DECAY,
)

# ================================================================================
# DEFAULTS
# ================================================================================
N_EPOCHS        = 1000
LR              = 1e-4
SEED            = 42
DATA_DIR        = ROOT_DIR / "data" / "preprocessed_data"
RESULTS_DIR     = ROOT_DIR / "results" / "training"

CSV_COLUMNS = [
    "epoch",
    "kl_g0_weight", "kl_u_weight",
    "train_loss_total", "train_loss_recon", "train_loss_kl_g0",
    "train_loss_kl_u",  "train_loss_cls",
    "val_loss_total",   "val_loss_recon",   "val_loss_kl_g0",
    "val_loss_kl_u",    "val_loss_cls",
    "lr",
]


# ================================================================================
# LOGGING
# ================================================================================
def setup_logging(results_dir: Path, run_name: str) -> logging.Logger:
    results_dir.mkdir(parents=True, exist_ok=True)
    log_path = results_dir / f"{run_name}.log"

    logger = logging.getLogger(run_name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(log_path)
    sh = logging.StreamHandler(sys.stdout)
    fmt = logging.Formatter("%(asctime)s | %(message)s")
    fh.setFormatter(fmt); sh.setFormatter(fmt)
    logger.addHandler(fh); logger.addHandler(sh)

    return logger


# ================================================================================
# KL ANNEALING WEIGHTS
# ================================================================================
def get_kl_weights(epoch: int) -> tuple[float, float]:
    """
    Compute KL annealing weights for g0 and u at a given epoch.

    KL_g0: linear ramp from 0 to 1 over KL_G0_ANNEAL_EPOCHS.
           Set KL_G0_ANNEAL_EPOCHS=0 in config to disable (weight=1.0 always).

    KL_u:  held at 0 for KL_U_DELAY_EPOCHS, then linear ramp over
           KL_U_ANNEAL_EPOCHS. Gives reconstruction time to stabilize
           before the control pathway is regularized.
           Set KL_U_ANNEAL_EPOCHS=0 in config to disable (weight=1.0 always).

    Args:
        epoch (int): Current training epoch (1-indexed)

    Returns:
        kl_g0_weight (float): weight on KL_g0 term, in [0, 1]
        kl_u_weight  (float): weight on KL_u term, in [0, 1]
    """
    kl_g0_weight = (
        max(0.0, min(1.0, (epoch - KL_G0_DELAY_EPOCHS) / KL_G0_ANNEAL_EPOCHS))
        if KL_G0_ANNEAL_EPOCHS > 0 else 1.0
    )
    kl_u_weight = (
        max(0.0, min(1.0, (epoch - KL_U_DELAY_EPOCHS) / KL_U_ANNEAL_EPOCHS))
        if KL_U_ANNEAL_EPOCHS > 0 else 1.0
    )
    return kl_g0_weight, kl_u_weight


# ================================================================================
# EPOCH
# ================================================================================
def run_epoch(
    model:           BRICK,
    subset,
    train:           bool,
    optimizer        = None,
    kl_g0_weight:    float = 1.0,
    kl_u_weight:     float = 1.0,
    apply_free_bits: bool  = True,
) -> dict:
    """
    Run one epoch over a dataset subset.

    Args:
        model            (BRICK):  The model
        subset:                    torch.utils.data.Subset
        train            (bool):   If True, update weights
        optimizer:                 AdamW (None in eval mode)
        kl_g0_weight     (float):  Annealing weight for KL_g0
        kl_u_weight      (float):  Annealing weight for KL_u
        apply_free_bits  (bool):   Apply free bits floor (True=train, False=val)

    Returns:
        dict of mean loss components over the epoch
    """
    model.train(train)

    totals = {
        "loss_total": 0.0, "loss_recon": 0.0,
        "loss_kl_g0": 0.0, "loss_kl_u":  0.0, "loss_cls": 0.0,
    }
    n = len(subset)

    for i in subset.indices:
        item  = subset.dataset[i]
        x     = item["x"]
        label = item["lifus_condition"]

        out  = model(
            x, label,
            kl_g0_weight=kl_g0_weight,
            kl_u_weight=kl_u_weight,
            apply_free_bits=apply_free_bits,
        )
        loss = out["losses"]["loss_total"]

        if train and optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for key in totals:
            totals[key] += out["losses"][key].item()

    return {k: v / n for k, v in totals.items()}


# ================================================================================
# MAIN TRAINING LOOP
# ================================================================================
def train(
    n_epochs:    int   = N_EPOCHS,
    use_control: bool  = True,
    use_ic:      bool  = True,
    run_name:    str   = "train",
):
    results_dir = RESULTS_DIR / run_name
    log = setup_logging(results_dir, run_name)

    log.info("=" * 60)
    log.info(f"BRICK Training -- {run_name}")
    log.info(f"N_ROIS={N_ROIS}, M={M}, H={H}, T={T_DATA}")
    log.info(f"Epochs={n_epochs}, LR={LR}, WeightDecay={WEIGHT_DECAY}")
    log.info(f"Patience={PATIENCE}, use_control={use_control}, use_ic={use_ic}")
    log.info(f"KL annealing: g0 over {KL_G0_ANNEAL_EPOCHS} epochs, "
             f"u delayed {KL_U_DELAY_EPOCHS} then over {KL_U_ANNEAL_EPOCHS} epochs")
    log.info("=" * 60)

    # --- Data ---
    log.info("Loading dataset...")
    ds = BRICKDataset(DATA_DIR)
    train_ds, val_ds, test_ds = split_dataset(ds, seed=SEED)
    log.info(
        f"Split: {len(train_ds)} train | {len(val_ds)} val | "
        f"{len(test_ds)} test items"
    )

    # Save split for reproducibility
    split_info = {
        "train": sorted(set(ds[i]["subject_id"] for i in train_ds.indices)),
        "val":   sorted(set(ds[i]["subject_id"] for i in val_ds.indices)),
        "test":  sorted(set(ds[i]["subject_id"] for i in test_ds.indices)),
        "seed":  SEED,
    }
    with open(results_dir / "split.json", "w") as f:
        json.dump(split_info, f, indent=2)

    # --- Model ---
    model = BRICK(use_control=use_control, use_ic=use_ic)
    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model parameters: {n_params:,}")

    # --- Optimizer and scheduler ---
    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs, eta_min=1e-6
    )

    # --- CSV ---
    csv_path = results_dir / "loss_history.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()

    # --- Training loop ---
    best_val_loss     = float("inf")
    epochs_no_improve = 0
    log.info("Starting training...")

    for epoch in range(1, n_epochs + 1):

        # KL annealing weights for this epoch
        kl_g0_weight, kl_u_weight = get_kl_weights(epoch)

        # Train — use free bits to prevent collapse
        train_losses = run_epoch(
            model, train_ds, train=True, optimizer=optimizer,
            kl_g0_weight=kl_g0_weight, kl_u_weight=kl_u_weight,
            apply_free_bits=True,
        )
        scheduler.step()

        # Validate — true ELBO, no free bits, full KL weights
        with torch.no_grad():
            val_losses = run_epoch(
                model, val_ds, train=False, optimizer=None,
                kl_g0_weight=1.0, kl_u_weight=1.0,
                apply_free_bits=False,
            )

        current_lr = scheduler.get_last_lr()[0]

        # --- Log to CSV ---
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

        # --- Log to terminal every 10 epochs ---
        if epoch % 10 == 0 or epoch == 1:
            log.info(
                f"Epoch {epoch:4d}/{n_epochs} | "
                f"kl_g0_w={kl_g0_weight:.2f} kl_u_w={kl_u_weight:.2f} | "
                f"train={train_losses['loss_total']:.4f} "
                f"(recon={train_losses['loss_recon']:.4f}, "
                f"kl_g0={train_losses['loss_kl_g0']:.4f}, "
                f"kl_u={train_losses['loss_kl_u']:.4f}, "
                f"cls={train_losses['loss_cls']:.4f}) | "
                f"val={val_losses['loss_total']:.4f} | "
                f"lr={current_lr:.2e}"
            )

        # --- Save best checkpoint ---
        val_total = val_losses["loss_total"]
        if val_total < best_val_loss:
            best_val_loss = val_total
            epochs_no_improve = 0
            torch.save({
                "epoch":                epoch,
                "model_state_dict":     model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss_total":       val_total,
                "train_loss_total":     train_losses["loss_total"],
                "use_control":          use_control,
                "use_ic":               use_ic,
                "h":                    model.h,
                "m":                    model.m,
            }, results_dir / "best_model.pt")
            log.info(f"  -> New best val loss: {val_total:.4f} (saved best_model.pt)")
        else:
            epochs_no_improve += 1

        # --- Early stopping ---
        if epochs_no_improve >= PATIENCE:
            log.info(
                f"Early stopping at epoch {epoch} -- "
                f"no improvement for {PATIENCE} epochs."
            )
            break

    # --- Save final checkpoint ---
    torch.save({
        "epoch":                epoch,
        "model_state_dict":     model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss_total":       val_losses["loss_total"],
        "train_loss_total":     train_losses["loss_total"],
        "use_control":          use_control,
        "use_ic":               use_ic,
        "h":                    model.h,
        "m":                    model.m,
    }, results_dir / "final_model.pt")

    log.info(f"Final model saved to {results_dir / 'final_model.pt'}")
    log.info(f"Best val loss: {best_val_loss:.4f}")
    log.info("Training complete.")

    return best_val_loss


# ================================================================================
# ENTRY POINT
# ================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train BRICK model")
    parser.add_argument("--epochs",     type=int,  default=N_EPOCHS)
    parser.add_argument("--no-control", action="store_true",
                        help="Ablation: disable control module")
    parser.add_argument("--no-ic",      action="store_true",
                        help="Ablation: disable initial condition module")
    parser.add_argument("--run-name",   type=str,  default=None,
                        help="Name for results subdirectory")
    args = parser.parse_args()

    use_control = not args.no_control
    use_ic      = not args.no_ic

    if args.run_name:
        run_name = args.run_name
    elif not use_control and not use_ic:
        run_name = "ablation_no_control_no_ic"
    elif not use_control:
        run_name = "ablation_no_control"
    elif not use_ic:
        run_name = "ablation_no_ic"
    else:
        run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    train(
        n_epochs=args.epochs,
        use_control=use_control,
        use_ic=use_ic,
        run_name=run_name,
    )