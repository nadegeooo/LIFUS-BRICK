"""
================================================================================
Seed-effects ROI comparison  —  reuses compare_pre_post.py's functions directly
================================================================================

Adapted from compare_batch_size.py, following the same pattern: does NOT call
compare_pre_post.main() and does NOT modify compare_pre_post.py. Imports the
individual functions it needs (model loading, K computation, ROI projection,
paired FDR tests, brain-space consistency check) and manages its own caching
under results/seed_effects/cache/, independent of compare_pre_post.py's and
compare_batch_size.py's output paths.

ASSUMPTION (edit find_checkpoint() if wrong): checkpoints live under
    results/training/sweep_3_seeds/sweep_{SEED_TYPE}_{value}/best_model_cls_preoverfit.pt
e.g. sweep_3_seeds/sweep_TRAIN_SEED_42/, sweep_3_seeds/sweep_SPLIT_SEED_123/, etc.

TRAIN_SEED and SPLIT_SEED are treated as two independent sweep axes (each
varied over {42, 123, 2024} while the other presumably held at a default),
NOT a full 3x3 grid. If it's actually a full grid, this script needs
restructuring -- flag that and it can be redone.

Produces, per target (vim, zi) x per seed_type (TRAIN_SEED, SPLIT_SEED), a
24-panel grid (one panel per ROI) showing decoder-projected mean delta-C
(post - pre) across seed values 42, 123, 2024, color-coded:

    green = FDR-significant           (paired_tests_per_roi, alpha=0.05)
    blue  = not FDR-significant, but
            >=15/19 subjects agree on the sign of the brain-space projection
            (check_per_subject_consistency_brain_space)
    red   = neither

Bar height uses the decoder-projected "delta" from paired_tests_per_roi (the
same quantity the FDR test itself was run on) -- NOT the quadratic-form
"mean_delta" from the consistency check, which lives in a different
projection and isn't comparable in magnitude. Consistency is used only for
color.

Usage:
    python analysis/compare_seed_effects.py
    python analysis/compare_seed_effects.py --force-recompute
"""

import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))  # so `import compare_pre_post` works

import compare_pre_post as cpp

SEED_TYPES = ["TRAIN_SEED", "SPLIT_SEED"]
SEED_VALUES = [42, 123, 2024]
TARGETS = ["vim", "zi"]
ALPHA = 0.05
CONSISTENCY_THRESHOLD = 15  # matches compare_pre_post's report_consistency_brain_space default

CACHE_DIR = ROOT_DIR / "results" / "seed_effects" / "cache"
OUT_DIR = ROOT_DIR / "results" / "seed_effects"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================================
# LOCATE CHECKPOINTS
# ================================================================================

def find_checkpoint(seed_type: str, seed_value: int) -> Path:
    """
    Looks under results/training/sweep_3_seeds/sweep_{SEED_TYPE}_{value}/ only.
    EDIT THIS if the actual folder naming under sweep_3_seeds differs.
    """
    pattern = f"sweep_3_seeds/sweep_{seed_type}_{seed_value}/best_model_cls_preoverfit.pt"
    candidates = sorted((ROOT_DIR / "results" / "training").glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for {seed_type}={seed_value} matching "
            f"results/training/{pattern}"
        )
    if len(candidates) > 1:
        print(f"  WARNING: multiple checkpoints found for {seed_type}={seed_value}: "
              f"{[str(c) for c in candidates]}. Using the most recently modified.")
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def cache_dir_for(checkpoint_path: Path, seed_type: str, seed_value: int) -> Path:
    run_id = f"{seed_type}_{seed_value}_{checkpoint_path.parent.name}"
    return CACHE_DIR / run_id


def has_cache(cdir: Path) -> bool:
    required = (
        [f"statistical_results_roi_{t}.csv" for t in TARGETS]
        + [f"consistency_brain_{t}.csv" for t in TARGETS]
    )
    return all((cdir / f).exists() for f in required)


# ================================================================================
# ANALYSIS (reuses compare_pre_post.py functions; does not touch its I/O)
# ================================================================================

def analyze_checkpoint(checkpoint_path: Path) -> dict:
    """Returns {target: (roi_df, consistency_df)} for TARGETS, computed fresh."""
    model = cpp.load_model(checkpoint_path)
    K, Lambda, W_bar_x = cpp.compute_K(model)
    roi_weights = cpp.compute_roi_projection_weights(W_bar_x)

    # Full C matrices (pre/post) per subject, needed for the brain-space
    # consistency check (it needs the full (96,96) matrix, not just diag(C)).
    subjects = cpp.load_all()
    cpp.verify_roi_consistency(subjects)
    C_dict = {}
    with torch.no_grad():
        for s in subjects:
            x_pre  = cpp.znorm(torch.tensor(s["mpre"],  dtype=torch.float32))
            x_post = cpp.znorm(torch.tensor(s["mpost"], dtype=torch.float32))
            key = (s["subject_id"], s["target"])
            C_dict[key] = {
                "pre":  model(x_pre)["C"],
                "post": model(x_post)["C"],
            }

    results = {}
    for target in TARGETS:
        sids, pre, post = cpp.extract_C_diagonals(model, target)
        if len(sids) < 2:
            print(f"  Skipping {target}: <2 subjects.")
            continue
        pre_roi  = cpp.project_to_roi(pre,  roi_weights)
        post_roi = cpp.project_to_roi(post, roi_weights)
        roi_df = cpp.paired_tests_per_roi(pre_roi, post_roi, cpp.TARGET_ROIS, ALPHA)
        consistency_df = cpp.check_per_subject_consistency_brain_space(
            C_dict, model, target=target
        )
        results[target] = (roi_df, consistency_df)

    return results


def ensure_analyzed(seed_type: str, seed_value: int, force: bool = False) -> Path:
    """Return the cache dir for this (seed_type, seed_value), computing + caching only if needed."""
    checkpoint_path = find_checkpoint(seed_type, seed_value)
    cdir = cache_dir_for(checkpoint_path, seed_type, seed_value)

    if not force and has_cache(cdir):
        print(f"{seed_type}={seed_value}: using cached results at {cdir}")
        return cdir

    print(f"{seed_type}={seed_value}: no cached results at {cdir}, analyzing "
          f"{checkpoint_path} ...")
    cdir.mkdir(parents=True, exist_ok=True)
    results = analyze_checkpoint(checkpoint_path)
    for target, (roi_df, consistency_df) in results.items():
        roi_df.to_csv(cdir / f"statistical_results_roi_{target}.csv", index=False)
        consistency_df.to_csv(cdir / f"consistency_brain_{target}.csv", index=False)
    print(f"  Cached to {cdir}")
    return cdir


# ================================================================================
# AGGREGATE
# ================================================================================

def load_seed_data(force: bool = False) -> pd.DataFrame:
    rows = []
    for seed_type in SEED_TYPES:
        for seed_value in SEED_VALUES:
            cdir = ensure_analyzed(seed_type, seed_value, force=force)
            for target in TARGETS:
                roi_path = cdir / f"statistical_results_roi_{target}.csv"
                cons_path = cdir / f"consistency_brain_{target}.csv"
                if not (roi_path.exists() and cons_path.exists()):
                    print(f"  WARNING: missing results for {seed_type}={seed_value}, "
                          f"target={target}; skipping.")
                    continue

                roi_df = pd.read_csv(roi_path)
                cons_df = pd.read_csv(cons_path).set_index("roi")

                for _, r in roi_df.iterrows():
                    roi = r["roi_name"]
                    if roi in cons_df.index:
                        n_consistent = int(cons_df.loc[roi, "n_consistent"])
                        n_subjects = int(cons_df.loc[roi, "n_positive"] + cons_df.loc[roi, "n_negative"])
                    else:
                        n_consistent, n_subjects = 0, np.nan

                    if bool(r["significant"]):
                        classification = "fdr_significant"
                    elif n_consistent >= CONSISTENCY_THRESHOLD:
                        classification = "consistency_threshold"
                    else:
                        classification = "neither"

                    rows.append({
                        "target":          target,
                        "seed_type":       seed_type,
                        "seed_value":      seed_value,
                        "roi":             roi,
                        "delta":           r["delta"],
                        "p_value_fdr":     r["p_value_fdr"],
                        "significant_fdr": bool(r["significant"]),
                        "n_consistent":    n_consistent,
                        "n_subjects":      n_subjects,
                        "classification":  classification,
                    })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "seed_effects_roi_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved aggregated summary: {out_csv}")
    return df


# ================================================================================
# PLOT
# ================================================================================

COLOR_MAP = {
    "fdr_significant":       "#2ca02c",   # green
    "consistency_threshold": "#1f77b4",   # blue
    "neither":                "#d62728",  # red
}


def plot_target_seed_type(df: pd.DataFrame, target: str, seed_type: str, ylim: tuple):
    sub = df[(df["target"] == target) & (df["seed_type"] == seed_type)]
    rois = list(cpp.TARGET_ROIS)
    n_rois = len(rois)
    ncols = 4
    nrows = int(np.ceil(n_rois / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_data = sub[sub["roi"] == roi].set_index("seed_value").reindex(SEED_VALUES)
        colors = [COLOR_MAP.get(c, "gray") for c in roi_data["classification"]]
        ax.bar([str(s) for s in SEED_VALUES], roi_data["delta"].values, color=colors)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(ylim)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx // ncols == nrows - 1:
            ax.set_xlabel(seed_type, fontsize=8)
        if idx % ncols == 0:
            ax.set_ylabel("\u0394C (decoder-projected)", fontsize=8)

    for idx in range(n_rois, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["fdr_significant"],
                      label="FDR significant"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["consistency_threshold"],
                      label=f">={CONSISTENCY_THRESHOLD}/19 consistent direction"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["neither"],
                      label="Neither"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=3, fontsize=9,
               bbox_to_anchor=(0.5, 1.02))
    fig.suptitle(f"\u0394C by {seed_type}, per ROI \u2014 {target.upper()}",
                fontsize=13, y=1.05)
    fig.tight_layout()

    out_path = OUT_DIR / f"roi_by_{seed_type.lower()}_{target}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# ENTRY POINT
# ================================================================================

def compute_shared_ylim(df: pd.DataFrame, pad_frac: float = 0.1) -> tuple:
    """Symmetric y-limit (centered on 0) covering every ROI/target/seed_type/seed_value,
    so all 4 output figures are directly comparable by eye."""
    max_abs = df["delta"].abs().max()
    if not np.isfinite(max_abs) or max_abs == 0:
        return (-1.0, 1.0)
    padded = max_abs * (1 + pad_frac)
    return (-padded, padded)


def main(force: bool = False):
    df = load_seed_data(force=force)
    ylim = compute_shared_ylim(df)
    for target in TARGETS:
        for seed_type in SEED_TYPES:
            plot_target_seed_type(df, target, seed_type, ylim=ylim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-analyze every seed value even if cached "
                             "results already exist under results/seed_effects/cache/.")
    args = parser.parse_args()
    main(force=args.force_recompute)