"""
analysis/compare_loso_K_spectra.py

Basis-free comparison of K's eigenvalue spectrum across LOSO folds, plus a
mode-index tracking view: for each fold, which original mode index landed
at each persistence rank. Rank itself is basis-free (just sorted |lambda|);
mode index is NOT basis-free (arbitrary per-fold eigensolver ordering), so
the index-heatmap is a qualitative "does the same index keep landing in the
same rank column" check, not a claim that indices are directly comparable
by construction the way ranks are.

Usage:
    python analysis/compare_loso_K_spectra.py
"""

import re
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from analysis.analysis_helper_functions import load_model, compute_K

LOSO_RUN_DIR = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2"
FIGURES_DIR  = ROOT_DIR / "results" / "loso" / "figures"

CHECKPOINT_NAME = "best_model_cls.pt"

NONCONVERGED_FOLDS = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}


def natural_fold_key(fold_dir_name: str):
    m = re.search(r"(\d+)$", fold_dir_name)
    return int(m.group(1)) if m else fold_dir_name


def discover_fold_dirs(run_dir: Path) -> list:
    return sorted(
        [d for d in run_dir.iterdir() if d.is_dir() and d.name.startswith("fold_")],
        key=lambda d: natural_fold_key(d.name),
    )


def load_fold_spectra(run_dir: Path) -> dict:
    """
    For each fold_sub-fuspdXX/ directory, load the checkpoint, compute K,
    and return {fold_id: (sorted_desc_mags, sorted_desc_orig_indices)}.
    orig_indices[r] = the raw eigensolver index (0..M-1) of the mode that
    landed at rank r (0 = most persistent) in this fold.
    """
    spectra = {}
    fold_dirs = discover_fold_dirs(run_dir)
    print(f"Found {len(fold_dirs)} fold directories")

    for fold_dir in fold_dirs:
        ckpt_path = fold_dir / CHECKPOINT_NAME
        if not ckpt_path.exists():
            print(f"  SKIP {fold_dir.name}: no {CHECKPOINT_NAME}")
            continue
        fold_id = fold_dir.name.replace("fold_", "")
        model = load_model(ckpt_path)
        _, Lambda, _ = compute_K(model)

        mag = np.abs(Lambda)
        order = np.argsort(mag)[::-1]          # indices, descending persistence
        mags_sorted = mag[order]
        orig_indices = order.copy()             # orig_indices[r] = mode # at rank r

        spectra[fold_id] = (mags_sorted, orig_indices)
        print(f"  Loaded {fold_id}: {len(mag)} modes, max|\u03bb|={mags_sorted[0]:.3f}")

    return spectra


def build_matrices(spectra: dict, fold_order: list = None):
    fold_ids = fold_order if fold_order is not None else sorted(
        spectra.keys(), key=natural_fold_key
    )
    M = len(next(iter(spectra.values()))[0])
    assert all(len(spectra[f][0]) == M for f in fold_ids), \
        "Folds have different numbers of modes -- check M is consistent across folds."

    mag_mat = np.stack([spectra[f][0] for f in fold_ids])
    idx_mat = np.stack([spectra[f][1] for f in fold_ids])
    return mag_mat, idx_mat, fold_ids


def _mark_nonconverged_labels(ax, fold_ids):
    for i, fid in enumerate(fold_ids):
        if fid in NONCONVERGED_FOLDS:
            ax.get_yticklabels()[i].set_color("red")
            ax.get_yticklabels()[i].set_fontweight("bold")
    legend_handle = mpatches.Patch(color="red", label="Flagged non-converged")
    ax.legend(handles=[legend_handle], loc="upper right",
              bbox_to_anchor=(1.0, 1.08), fontsize=8, frameon=False)


def plot_rank_heatmap(mag_mat: np.ndarray, fold_ids: list, out_path: Path):
    """|lambda| by rank -- basis-free, autoscaled + high-contrast colormap."""
    n_folds, M = mag_mat.shape
    fig, ax = plt.subplots(figsize=(14, max(4, n_folds * 0.35)))

    vmin, vmax = mag_mat.min(), mag_mat.max()
    im = ax.imshow(mag_mat, aspect="auto", cmap="turbo", vmin=vmin, vmax=vmax,
                   interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="|\u03bb|")

    ax.set_xlabel("Rank (1 = most persistent)")
    ax.set_ylabel("LOSO fold")
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels(fold_ids, fontsize=8)
    _mark_nonconverged_labels(ax, fold_ids)

    ax.set_title(
        "Eigenvalue persistence spectrum by rank, across LOSO folds\n"
        "(fold order = native LOSO numbering; color autoscaled to data range)",
        fontsize=11
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_mode_index_heatmap(idx_mat: np.ndarray, fold_ids: list, M: int, out_path: Path):
    """
    Which original mode index landed at each rank, per fold. Color = mode
    index (0..M-1), qualitative/cyclic colormap so adjacent indices are NOT
    visually similar -- what you're looking for is the SAME color reappearing
    in the SAME column across rows, not a smooth left-right gradient.
    """
    n_folds, n_ranks = idx_mat.shape
    fig, ax = plt.subplots(figsize=(16, max(4, n_folds * 0.4)))

    im = ax.imshow(idx_mat, aspect="auto", cmap="hsv", vmin=0, vmax=M - 1,
                   interpolation="nearest")
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.02, label="Original mode index")

    ax.set_xlabel("Rank (left = most persistent)")
    ax.set_ylabel("LOSO fold")
    ax.set_yticks(range(n_folds))
    ax.set_yticklabels(fold_ids, fontsize=8)
    _mark_nonconverged_labels(ax, fold_ids)

    # annotate mode index in each cell -- with M~96 and 19 folds this is dense,
    # so only annotate the top ranks where it's most readable/most relevant
    n_annot = min(20, n_ranks)
    for i in range(n_folds):
        for j in range(n_annot):
            ax.text(j, i, str(idx_mat[i, j]), ha="center", va="center",
                    fontsize=5.5, color="black")

    ax.set_title(
        "Original mode index by persistence rank, across LOSO folds\n"
        "(look for the same color/number repeating down a column = "
        "same mode index consistently ranked similarly)",
        fontsize=11
    )
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def compute_mode_persistence_stats(idx_mat: np.ndarray, mag_mat: np.ndarray,
                                    fold_ids: list, exclude_folds: set = None) -> pd.DataFrame:
    """
    For each original mode index, gather its |lambda| value from every fold
    (found via idx_mat, since a mode's RANK can shift fold-to-fold even when
    its identity is stable) and compute mean/std across folds.

    exclude_folds: fold IDs to leave out of the stats (e.g. NONCONVERGED_FOLDS)
    -- included is fine, but contaminates the "reliable model" estimate, so
    default behavior here is to exclude them.
    """
    exclude_folds = exclude_folds or set()
    keep = [i for i, fid in enumerate(fold_ids) if fid not in exclude_folds]
    n_kept = len(keep)

    M = idx_mat.shape[1]
    # per-fold inverse map: mode_index -> |lambda|, for each kept fold
    per_mode_vals = {m: [] for m in range(M)}
    for i in keep:
        for rank in range(M):
            mode_idx = idx_mat[i, rank]
            per_mode_vals[mode_idx].append(mag_mat[i, rank])

    rows = []
    for mode_idx, vals in per_mode_vals.items():
        vals = np.array(vals)
        rows.append({
            "mode_index": mode_idx,
            "n_folds": len(vals),          # should equal n_kept unless a fold is missing this index
            "mean_mag": vals.mean(),
            "std_mag": vals.std(ddof=1) if len(vals) > 1 else np.nan,
            "min_mag": vals.min(),
            "max_mag": vals.max(),
        })

    df = pd.DataFrame(rows).sort_values("mean_mag", ascending=False).reset_index(drop=True)
    df["mean_rank"] = df.index + 1   # rank position by mean persistence, 1-indexed
    print(f"Computed stats across {n_kept} converged folds "
          f"(excluded: {sorted(exclude_folds) if exclude_folds else 'none'})")
    return df


def plot_mode_persistence_errorbar(df: pd.DataFrame, out_path: Path, top_k: int = 30):
    """Mean +/- std of |lambda| for the top_k most persistent modes (by mean),
    x-axis = original mode index (labeled), so you can see both the value
    and which specific mode it is."""
    top = df.head(top_k)
    fig, ax = plt.subplots(figsize=(max(10, top_k * 0.4), 5))

    x = range(len(top))
    ax.errorbar(x, top["mean_mag"], yerr=top["std_mag"], fmt="o", color="steelblue",
                ecolor="lightsteelblue", elinewidth=2, capsize=3, markersize=5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"M{m}" for m in top["mode_index"]], rotation=90, fontsize=8)
    ax.set_ylabel("|\u03bb|  (mean \u00b1 std across converged folds)")
    ax.set_xlabel("Original mode index (ordered by mean persistence)")
    ax.set_title(f"Top {top_k} modes: persistence consistency across LOSO folds\n"
                 "(narrow error bars = reliably learned; wide = fold-idiosyncratic)",
                 fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def compute_fold_persistence_summary(mag_mat: np.ndarray, fold_ids: list,
                                      top_k: int = 20) -> pd.DataFrame:
    """
    Per fold: how persistent are the dominant modes, on average?
    mag_mat is already rank-sorted per fold (descending |lambda|), so
    mag_mat[i, :top_k] is exactly "this fold's top-k most persistent modes."
    """
    rows = []
    for i, fid in enumerate(fold_ids):
        top_vals = mag_mat[i, :top_k]
        rows.append({
            "fold_id": fid,
            "mean_top_k_persistence": top_vals.mean(),
            "max_persistence": mag_mat[i, 0],          # single most persistent mode
            "spectral_radius": mag_mat[i, 0],           # same thing, named per convention
            "mean_all_persistence": mag_mat[i].mean(),  # whole spectrum, not just top-k
        })
    return pd.DataFrame(rows)


def load_fold_losses(run_dir: Path, fold_ids: list) -> pd.DataFrame:
    """
    Reads loss_history.csv per fold, takes the best (min) val_loss_recon and
    val_loss_cls achieved during training. Adjust column names below to match
    your actual loss_history.csv schema.
    """
    rows = []
    for fold_dir in discover_fold_dirs(run_dir):
        fid = fold_dir.name.replace("fold_", "")
        csv_path = fold_dir / "loss_history.csv"
        if not csv_path.exists() or fid not in fold_ids:
            continue
        hist = pd.read_csv(csv_path)
        rows.append({
            "fold_id": fid,
            "best_val_loss_recon": hist["val_loss_recon"].min(),   # <- verify col name
            "best_val_loss_cls":   hist["val_loss_cls"].min(),     # <- verify col name
        })
    return pd.DataFrame(rows)


def plot_persistence_vs_loss(persistence_df: pd.DataFrame, loss_df: pd.DataFrame,
                              out_path: Path, persistence_col: str = "mean_top_k_persistence"):
    from scipy.stats import spearmanr

    merged = persistence_df.merge(loss_df, on="fold_id")
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, loss_col, title in zip(
        axes, ["best_val_loss_recon", "best_val_loss_cls"],
        ["Reconstruction loss", "Classification loss"]
    ):
        colors = ["red" if fid in NONCONVERGED_FOLDS else "steelblue"
                  for fid in merged["fold_id"]]
        ax.scatter(merged[persistence_col], merged[loss_col], c=colors, s=60)

        # label every point, not just flagged ones
        for _, row in merged.iterrows():
            ax.annotate(row["fold_id"], (row[persistence_col], row[loss_col]),
                        fontsize=6.5, xytext=(4, 4), textcoords="offset points")

        rho, p = spearmanr(merged[persistence_col], merged[loss_col])
        ax.set_xlabel(f"{persistence_col} (top-20 modes)")
        ax.set_ylabel(title)
        ax.set_title(f"{title} vs. mode persistence\nSpearman \u03c1={rho:.2f}, p={p:.3f}",
                     fontsize=10)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def main():
    spectra = load_fold_spectra(LOSO_RUN_DIR)
    mag_mat, idx_mat, fold_ids = build_matrices(spectra)
    M = mag_mat.shape[1]

    plot_rank_heatmap(mag_mat, fold_ids, FIGURES_DIR / "loso_spectrum_heatmap.png")
    plot_mode_index_heatmap(idx_mat, fold_ids, M, FIGURES_DIR / "loso_mode_index_heatmap.png")

    stats_df = compute_mode_persistence_stats(idx_mat, mag_mat, fold_ids,
                                               exclude_folds=NONCONVERGED_FOLDS)
    stats_path = FIGURES_DIR.parent / "loso_mode_persistence_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"Saved {stats_path}")

    plot_mode_persistence_errorbar(stats_df, FIGURES_DIR / "loso_mode_persistence_errorbar.png")

    loss_df = load_fold_losses(LOSO_RUN_DIR, fold_ids)

    persistence_df = compute_fold_persistence_summary(mag_mat, fold_ids, top_k=20)
    plot_persistence_vs_loss(persistence_df, loss_df,
                              FIGURES_DIR / "loso_persistence_vs_loss.png")

if __name__ == "__main__":
    main()