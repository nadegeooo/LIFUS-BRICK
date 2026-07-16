# training/ablation_study.py
import pandas as pd
import matplotlib.pyplot as plt
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from training.train import train

ROOT_DIR = Path(__file__).resolve().parent.parent

ABLATIONS = {
    "full":              {"use_control": True,  "use_ic": True},
    "no_control":        {"use_control": False, "use_ic": True},
    "no_ic":             {"use_control": True,  "use_ic": False},
    "no_control_no_ic":  {"use_control": False, "use_ic": False},
}
METRICS = [
    ("loss_total",  "Total Loss"),
    ("loss_recon",  "Reconstruction Loss"),
    ("loss_kl_g0",  "KL g0"),
    ("loss_kl_u",   "KL u"),
    ("loss_cls",    "Classification Loss"),
]


def run_ablations(ablation_name: str, n_epochs: int = 1000, batch_size: int = None):
    training_dir = ROOT_DIR / "results" / "training" / ablation_name
    figures_dir  = ROOT_DIR / "results" / "figures" / ablation_name

    for name, kwargs in ABLATIONS.items():
        print(f"\n--- Running ablation: {name} (batch_size={batch_size}) ---")
        train(
            n_epochs=n_epochs,
            run_name=f"{ablation_name}/ablation_{name}",
            batch_size=batch_size,
            **kwargs,
        )
    plot_ablations(ablation_name, training_dir, figures_dir)


def plot_ablations(ablation_name: str, training_dir: Path, figures_dir: Path):
    figures_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(METRICS), 2, figsize=(14, 4 * len(METRICS)))
    fig.suptitle(f"BRICK Ablation Study — {ablation_name}", fontsize=14)
    for row, (metric, title) in enumerate(METRICS):
        ax_train, ax_val = axes[row]
        for name in ABLATIONS:
            csv_path = training_dir / f"ablation_{name}" / "loss_history.csv"
            if not csv_path.exists():
                continue
            df = pd.read_csv(csv_path)
            ax_train.plot(df["epoch"], df[f"train_{metric}"], label=name)
            ax_val.plot(df["epoch"], df[f"val_{metric}"], label=name)
        ax_train.set_title(f"{title} — train")
        ax_val.set_title(f"{title} — val")
        ax_train.legend(); ax_val.legend()
        ax_train.set_xlabel("epoch"); ax_val.set_xlabel("epoch")
    plt.tight_layout()
    out = figures_dir / "ablation_results.png"
    plt.savefig(out, dpi=150)
    print(f"\nPlot saved to {out}")

    # --- Comparison table ---
    lines = []
    lines.append(f"{'Condition':<25} {'Best Val Recon':<18} {'Best Val Cls':<15} {'Best Val Total':<15}")
    lines.append("-" * 75)
    for name in ABLATIONS:
        csv_path = training_dir / f"ablation_{name}" / "loss_history.csv"
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        best_recon = df["val_loss_recon"].min()
        best_cls   = df["val_loss_cls"].min()
        best_total = df["val_loss_total"].min()
        lines.append(f"{name:<25} {best_recon:<18.4f} {best_cls:<15.4f} {best_total:<15.4f}")
    output = "\n".join(lines)
    print("\n" + output)
    table_path = figures_dir / "ablation_results.txt"
    table_path.write_text(output)
    print(f"Table saved to {table_path}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--ablation-name", type=str, required=True,
                        help="Folder name to group this ablation study under, e.g. ablation_1")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Override BATCH_SIZE for all runs in this ablation study")
    parser.add_argument("--plot-only", action="store_true",
                        help="Skip training, just plot existing results for --ablation-name")
    args = parser.parse_args()

    if args.plot_only:
        training_dir = ROOT_DIR / "results" / "training" / args.ablation_name
        figures_dir  = ROOT_DIR / "results" / "figures" / args.ablation_name
        plot_ablations(args.ablation_name, training_dir, figures_dir)
    else:
        run_ablations(args.ablation_name, n_epochs=args.epochs, batch_size=args.batch_size)