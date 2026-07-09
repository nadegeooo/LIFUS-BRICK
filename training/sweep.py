# training/sweep.py
import pandas as pd
import matplotlib.pyplot as plt

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from training.train import train

ROOT_DIR     = Path(__file__).resolve().parent.parent

SWEEP_NAME = "sweep_1"   # <-- change this each time

TRAINING_DIR = ROOT_DIR / "results" / "training" / SWEEP_NAME
FIGURES_DIR  = ROOT_DIR / "results" / "figures" / SWEEP_NAME

BASELINE = {
    "LAMBDA_NOISE": 0.01,
    "WEIGHT_DECAY": 0.05,
    "BATCH_SIZE":   12,
    "BETA":         0.1,
    "EPSILON":      1.0,
}

SWEEP = {
    "LAMBDA_NOISE": [0.001, 0.01, 0.1],
    "WEIGHT_DECAY": [1e-2, 0.05, 1e-3],
    "BATCH_SIZE":   [4, 8, 12, 16],
    "BETA":         [0.01, 0.05, 0.1],
    "EPSILON":      [0.5, 1.0, 2.0],
}

PARAM_MAP = {
    "LAMBDA_NOISE": "lambda_noise",
    "WEIGHT_DECAY": "weight_decay",
    "BATCH_SIZE":   "batch_size",
    "BETA":         "beta",
    "EPSILON":      "epsilon",
}

def run_sweep():
    for param, values in SWEEP.items():
        for val in values:
            kwargs = {PARAM_MAP[p]: BASELINE[p] for p in BASELINE}
            kwargs[PARAM_MAP[param]] = val
            run_name = f"sweep_{param}_{val}"
            train(n_epochs=200, run_name=f"{SWEEP_NAME}/{run_name}", **kwargs)

    plot_sweep()

def plot_sweep():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(SWEEP), 2, figsize=(12, 4 * len(SWEEP)))

    for row, (param, values) in enumerate(SWEEP.items()):
        ax_recon, ax_cls = axes[row]
        for val in values:
            run_name = f"sweep_{param}_{val}"
            csv_path = TRAINING_DIR / run_name / "loss_history.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ax_recon.plot(df["epoch"], df["val_loss_recon"], label=f"{val}")
            ax_cls.plot(df["epoch"], df["val_loss_cls"], label=f"{val}")
        ax_recon.set_title(f"{param} — val recon loss")
        ax_cls.set_title(f"{param} — val cls loss")
        ax_recon.legend(); ax_cls.legend()
        ax_recon.set_xlabel("epoch"); ax_cls.set_xlabel("epoch")

    plt.tight_layout()
    out = FIGURES_DIR / "sweep_results.png"
    plt.savefig(out, dpi=150)
    print(f"Plot saved to {out}")

    lines = []
    lines.append(f"{'Param':<20} {'Value':<10} {'Best Val Recon':<18} {'Best Val Cls':<15}")
    lines.append("-" * 65)
    for param, values in SWEEP.items():
        for val in values:
            run_name = f"sweep_{param}_{val}"
            csv_path = TRAINING_DIR / run_name / "loss_history.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            best_recon = df["val_loss_recon"].min()
            best_cls   = df["val_loss_cls"].min()
            lines.append(f"{param:<20} {str(val):<10} {best_recon:<18.4f} {best_cls:<15.4f}")
        lines.append("")

    output = "\n".join(lines)
    print(output)
    table_path = FIGURES_DIR / "sweep_results.txt"
    table_path.write_text(output)
    print(f"Table saved to {table_path}")

if __name__ == "__main__":
    run_sweep()