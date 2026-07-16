"""
Pareto frontier (reconstruction vs classification) from an existing OAT sweep.

Reads the loss_history.csv files your sweep already produced and places one
point per run at the SAME epoch its best_model_cls checkpoint was
written (the model you actually analyze), not at the independent minima of
each column (which are a phantom operating point at two different epochs).

Recon axis is raw MSE = val_loss_recon * 2 * lambda_noise, so runs with
different lambda_noise are comparable. cls axis is raw cross-entropy.

Only the beta and lambda arms trade recon for cls, so only those define the
frontier. Batch/epsilon/wd runs are context, not frontier axes.

Usage:
    python pareto_frontier.py --sweep-dir results/training/sweep_1
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Baseline held for every non-swept parameter (from your sweep.py BASELINE).
BASELINE = {"LAMBDA_NOISE": 0.01, "BETA": 0.05, "EPSILON": 1.0,
            "BATCH_SIZE": 4, "WEIGHT_DECAY": 0.05}

# Which arms trace the recon<->cls tradeoff. Others are plotted as context.
FRONTIER_ARMS = {"BETA", "LAMBDA_NOISE"}


def parse_run(run_name: str):
    """
    'sweep_BETA_0.1' -> ('BETA', 0.1). Returns (param, value) or (None, None)
    if the directory name doesn't match the sweep_<PARAM>_<value> pattern.
    """
    if not run_name.startswith("sweep_"):
        return None, None
    body = run_name[len("sweep_"):]
    for param in BASELINE:                       # match longest param names first
        if body.startswith(param + "_"):
            try:
                return param, float(body[len(param) + 1:])
            except ValueError:
                return param, None
    return None, None


def lambda_for_run(param: str, value: float) -> float:
    """lambda_noise actually used by this run."""
    return value if param == "LAMBDA_NOISE" else BASELINE["LAMBDA_NOISE"]


def checkpoint_row(df: pd.DataFrame):
    """
    Replays train.py's cls rule: the checkpoint is (re)written each
    epoch where BOTH val and train cls loss improve on their running best. We
    return the LAST such row -- the state saved to best_model_cls.pt.
    """
    best_val = best_train = np.inf
    row = None
    for _, r in df.iterrows():
        if r["val_loss_cls"] < best_val and r["train_loss_cls"] < best_train:
            best_val, best_train = r["val_loss_cls"], r["train_loss_cls"]
            row = r
    return row


def collect_points(sweep_dir: Path) -> pd.DataFrame:
    rows = []
    for run_dir in sorted(p for p in sweep_dir.iterdir() if p.is_dir()):
        csv = run_dir / "loss_history.csv"
        if not csv.exists():
            continue
        param, value = parse_run(run_dir.name)
        if param is None:
            continue
        df = pd.read_csv(csv)
        r = checkpoint_row(df)
        if r is None:
            print(f"  skip {run_dir.name}: no cls checkpoint in log")
            continue
        lam = lambda_for_run(param, value)
        rows.append({
            "run":   run_dir.name,
            "param": param,
            "value": value,
            "epoch": int(r["epoch"]),
            "mse":   float(r["val_loss_recon"]) * 2.0 * lam,   # raw MSE
            "cls":   float(r["val_loss_cls"]),                 # raw cross-entropy
            "arm":   param in FRONTIER_ARMS,
        })
    return pd.DataFrame(rows)


def pareto_front(pts: pd.DataFrame) -> pd.DataFrame:
    """Non-dominated set: minimise both mse and cls (lower-left is better)."""
    keep = []
    for i, a in pts.iterrows():
        dominated = any(
            (b["mse"] <= a["mse"] and b["cls"] <= a["cls"]) and
            (b["mse"] <  a["mse"] or  b["cls"] <  a["cls"])
            for _, b in pts.iterrows()
        )
        if not dominated:
            keep.append(i)
    return pts.loc[keep].sort_values("mse")


def main(sweep_dir: Path, out_path: Path):
    pts = collect_points(sweep_dir)
    if pts.empty:
        print("No usable runs found."); return

    arm_pts = pts[pts["arm"]].copy()
    front = pareto_front(arm_pts)

    fig, ax = plt.subplots(figsize=(7, 6))

    # Context points (batch/epsilon/wd) -- not on the frontier.
    ctx = pts[~pts["arm"]]
    ax.scatter(ctx["mse"], ctx["cls"], c="lightgrey", s=40,
               label="other params (context)", zorder=1)

    # Tradeoff-arm points, coloured by which knob moved.
    colors = {"BETA": "#d62728", "LAMBDA_NOISE": "#1f77b4"}
    for param, g in arm_pts.groupby("param"):
        ax.scatter(g["mse"], g["cls"], c=colors.get(param, "black"),
                   s=70, label=param, zorder=3)
        for _, r in g.iterrows():
            ax.annotate(f"{r['value']:g}", (r["mse"], r["cls"]),
                        textcoords="offset points", xytext=(6, 4), fontsize=8)

    # Frontier line.
    ax.plot(front["mse"], front["cls"], "k--", lw=1.2,
            label="Pareto frontier", zorder=2)

    ax.set_xlabel("reconstruction: raw MSE  (val_loss_recon x 2 x lambda)")
    ax.set_ylabel("classification: val cross-entropy")
    ax.set_title("Recon vs cls tradeoff\n(lower-left is better; label = swept value)")
    ax.axhline(np.log(2), color="green", ls=":", lw=1,
               label="chance cls (ln2 ~ 0.693)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")

    print("\nFrontier points (non-dominated):")
    print(front[["param", "value", "epoch", "mse", "cls"]].to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep-dir", type=Path, required=True,
                    help="e.g. results/training/sweep_1")
    ap.add_argument("--out", type=Path, default=Path("pareto_frontier.png"))
    args = ap.parse_args()
    main(args.sweep_dir, args.out)