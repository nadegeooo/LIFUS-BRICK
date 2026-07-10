"""
================================================================================
Sweep Analysis Script
================================================================================
For a given sweep (e.g. "sweep_1", "sweep_2"), this script:
  1. Plots per-run training curves (total, recon, KL_g0, KL_u, cls) with
     dotted vertical lines marking:
        - best epoch by val total loss           (best_model.pt)               -> red
        - best epoch by joint val/train cls loss  (best_model_cls_preoverfit.pt) -> purple
     Saved into each run's own results folder.
  2. Builds a sweep_summary table (best val total/recon/cls loss per run)
     saved to results/training/<SWEEP_NAME>/sweep_summary.csv and .txt

Usage:
    python training/sweep_analysis.py --sweep-name sweep_1
    python training/sweep_analysis.py --sweep-name sweep_2
================================================================================
"""

import sys
import csv
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))


# ================================================================================
# HELPERS
# ================================================================================
def list_runs(results_root: Path, subfolder: str = None) -> list:
    """List all runs in results/training or a named subfolder."""
    search_root = results_root / subfolder if subfolder else results_root
    csv_files = sorted(search_root.glob("*/loss_history.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No loss_history.csv found in {search_root}")
    print(f"Available runs in '{subfolder or 'root'}':")
    for i, f in enumerate(csv_files):
        print(f"  [{i+1}] {f.parent.name}")
    return csv_files


def load_csv(csv_path: Path) -> dict:
    """Load loss history CSV into lists."""
    data = {k: [] for k in [
        "epochs",
        "train_total", "val_total",
        "train_recon", "val_recon",
        "train_kl_g0", "val_kl_g0",
        "train_kl_u",  "val_kl_u",
        "train_cls",   "val_cls",
    ]}
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            data["epochs"].append(int(row["epoch"]))
            data["train_total"].append(float(row["train_loss_total"]))
            data["val_total"].append(float(row["val_loss_total"]))
            data["train_recon"].append(float(row["train_loss_recon"]))
            data["val_recon"].append(float(row["val_loss_recon"]))
            data["train_kl_g0"].append(float(row["train_loss_kl_g0"]))
            data["val_kl_g0"].append(float(row["val_loss_kl_g0"]))
            data["train_kl_u"].append(float(row["train_loss_kl_u"]))
            data["val_kl_u"].append(float(row["val_loss_kl_u"]))
            data["train_cls"].append(float(row["train_loss_cls"]))
            data["val_cls"].append(float(row["val_loss_cls"]))
    return data


def load_best_epoch(csv_path: Path, checkpoint_name: str = "best_model.pt") -> int | None:
    """
    Load the epoch recorded in a given checkpoint file in the same directory
    as csv_path. Generalized from the original best_model.pt-only version so
    it can also load best_model_cls_preoverfit.pt (or any other checkpoint
    that stores an 'epoch' key).
    """
    checkpoint_path = csv_path.parent / checkpoint_name
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        best_epoch = checkpoint["epoch"]
        print(f"  {checkpoint_name} saved at epoch {best_epoch}")
        return best_epoch
    print(f"  No {checkpoint_name} found.")
    return None


def load_val_metrics_at_checkpoint(csv_path: Path, checkpoint_name: str = "best_model_cls_preoverfit.pt") -> dict | None:
    """
    Return val_total, val_recon, val_cls all taken from the SAME epoch —
    the epoch recorded in the given checkpoint file. This ensures the three
    numbers describe one actual saved model, not three independently-best
    epochs that may never have co-occurred.

    Returns None if the checkpoint doesn't exist for this run.
    """
    epoch = load_best_epoch(csv_path, checkpoint_name)
    if epoch is None:
        return None

    data = load_csv(csv_path)
    if epoch not in data["epochs"]:
        print(f"  [warn] epoch {epoch} from {checkpoint_name} not found in loss_history.csv for {csv_path.parent.name}")
        return None

    idx = data["epochs"].index(epoch)
    return {
        "epoch":     epoch,
        "val_total": data["val_total"][idx],
        "val_recon": data["val_recon"][idx],
        "val_cls":   data["val_cls"][idx],
    }


# ================================================================================
# PLOTTING — single run
# ================================================================================
def plot_curves(data: dict, best_epoch_total: int | None,
                 best_epoch_cls_preoverfit: int | None,
                 run_name: str, out_path: Path):
    """Plot all loss curves for one run and save to out_path."""
    epochs = data["epochs"]

    def plot(ax, title, train, val=None):
        ax.plot(epochs, train, label="train", linewidth=1.5)
        if val is not None:
            ax.plot(epochs, val, label="val", linewidth=1.5, linestyle="--")
        if best_epoch_total is not None:
            ax.axvline(x=best_epoch_total, color="red", linestyle=":", linewidth=1.5,
                       label=f"best total (ep {best_epoch_total})")
        if best_epoch_cls_preoverfit is not None:
            ax.axvline(x=best_epoch_cls_preoverfit, color="purple", linestyle=":", linewidth=1.5,
                       label=f"best cls preoverfit (ep {best_epoch_cls_preoverfit})")
        ax.set_title(title)
        ax.set_xlabel("Epoch")
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    fig.suptitle(f"BRICK Training Loss Curves\n{run_name}", fontsize=13)
    plot(axes[0, 0], "Total Loss",          data["train_total"], data["val_total"])
    plot(axes[0, 1], "Reconstruction Loss", data["train_recon"], data["val_recon"])
    plot(axes[0, 2], "KL g0",               data["train_kl_g0"], data["val_kl_g0"])
    plot(axes[1, 0], "KL u",                data["train_kl_u"],  data["val_kl_u"])
    plot(axes[1, 1], "Classification Loss", data["train_cls"],   data["val_cls"])
    axes[1, 2].axis("off")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved plot to {out_path}")


# ================================================================================
# SUMMARY TABLE — across all runs in the sweep
# ================================================================================
def build_sweep_summary(csv_files: list, sweep_dir: Path):
    rows = []
    skipped = []
    for csv_path in csv_files:
        metrics = load_val_metrics_at_checkpoint(csv_path, "best_model_cls_preoverfit.pt")
        if metrics is None:
            skipped.append(csv_path.parent.name)
            continue
        rows.append({
            "run_name": csv_path.parent.name,
            "epoch":    metrics["epoch"],
            "val_total": metrics["val_total"],
            "val_recon": metrics["val_recon"],
            "val_cls":   metrics["val_cls"],
        })

    rows.sort(key=lambda r: r["run_name"])

    if skipped:
        print(f"\n[warn] Skipped {len(skipped)} run(s) with no best_model_cls_preoverfit.pt: {skipped}")

    # --- text table ---
    lines = []
    header = f"{'Run':<35} {'Epoch':<7} {'Val Total':<14} {'Val Recon':<14} {'Val Cls':<14}"
    lines.append(header)
    lines.append("-" * len(header))
    for r in rows:
        lines.append(
            f"{r['run_name']:<35} {r['epoch']:<7} {r['val_total']:<14.4f} "
            f"{r['val_recon']:<14.4f} {r['val_cls']:<14.4f}"
        )
    text_output = "\n".join(lines)
    print("\n" + text_output)

    txt_path = sweep_dir / "sweep_summary.txt"
    txt_path.write_text(text_output)
    print(f"\nSummary table (txt) saved to {txt_path}")

    csv_out_path = sweep_dir / "sweep_summary.csv"
    with open(csv_out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["run_name", "epoch", "val_total", "val_recon", "val_cls"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Summary table (csv) saved to {csv_out_path}")


# ================================================================================
# MAIN
# ================================================================================
def main():
    parser = argparse.ArgumentParser(description="Analyze and plot a BRICK hyperparameter sweep")
    parser.add_argument("--sweep-name", type=str, required=True,
                        help="Name of the sweep subfolder under results/training/, e.g. sweep_1")
    args = parser.parse_args()

    results_root = ROOT_DIR / "results" / "training"
    sweep_dir = results_root / args.sweep_name

    csv_files = list_runs(results_root, subfolder=args.sweep_name)
    print(f"\nFound {len(csv_files)} runs under {sweep_dir}")

    for csv_path in csv_files:
        run_dir = csv_path.parent
        print(f"\nProcessing: {run_dir.name}")

        data = load_csv(csv_path)
        best_epoch_total = load_best_epoch(csv_path, "best_model.pt")
        best_epoch_cls_preoverfit = load_best_epoch(csv_path, "best_model_cls_preoverfit.pt")

        out_path = run_dir / f"training_curves_{run_dir.name}.png"
        plot_curves(data, best_epoch_total, best_epoch_cls_preoverfit, run_dir.name, out_path)

    build_sweep_summary(csv_files, sweep_dir)


if __name__ == "__main__":
    main()