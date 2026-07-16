"""
================================================================================
Batch-size ROI comparison  —  reuses compare_pre_post.py's functions directly
================================================================================

Does NOT call compare_pre_post.main() and does NOT modify compare_pre_post.py.
That file's RESULTS_DIR/FIGURES_DIR/FINAL_MODEL_PATH are fixed constants (by
design, per the project owner) and main() always writes to the same path
regardless of which checkpoint it loaded -- calling it once per batch size
would overwrite its own output each time. Instead, this script imports the
individual functions it needs from compare_pre_post.py (model loading, K
computation, ROI projection, paired FDR tests, brain-space consistency check)
and manages its own caching under results/batch_size/cache/, independent of
compare_pre_post.py's output paths.

For each batch size in BATCH_SIZES, locates best_model_cls.pt
under results/training/sweep_*/sweep_BATCH_SIZE_{n}/. If cached results
already exist for that checkpoint under results/batch_size/cache/, loads them
directly. Otherwise runs the analysis once and caches it.

Produces, per target (vim, zi), a 24-panel grid (one panel per ROI) showing
decoder-projected mean ΔC (post - pre) across batch sizes 1,2,3,4,8,12,16,
color-coded:

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

Both output figures (vim, zi) share the same symmetric y-axis, computed from
the global max |delta| across both targets and all batch sizes, so the two
plots are directly comparable by eye.

IMPORTANT: batch_size=1 here means results/training/sweep_2/sweep_BATCH_SIZE_1,
NOT results/training/ablation_2_batch_size_1 -- the latter looks like it's
from the ablation study (different config), not a same-conditions batch-size
sweep, so it's deliberately excluded. Edit find_checkpoint()'s glob pattern
if that assumption is wrong.

Usage:
    python analysis/compare_batch_size.py
    python analysis/compare_batch_size.py --force-recompute
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

BATCH_SIZES = [1, 2, 3, 4, 8, 12, 16]
TARGETS = ["vim", "zi"]
ALPHA = 0.05
CONSISTENCY_THRESHOLD = 15  # matches compare_pre_post's report_consistency_brain_space default

CACHE_DIR = ROOT_DIR / "results" / "batch_size" / "cache"
OUT_DIR = ROOT_DIR / "results" / "batch_size"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# ================================================================================
# LOCATE CHECKPOINTS
# ================================================================================

def find_checkpoint(batch_size: int) -> Path:
    """
    Looks under results/training/sweep_*/sweep_BATCH_SIZE_{n}/ only.
    Deliberately does NOT match ablation_2_batch_size_1 or any other
    ablation_* folder -- those are a different experiment type.
    """
    pattern = f"sweep_*/sweep_BATCH_SIZE_{batch_size}/best_model_cls.pt"
    candidates = sorted((ROOT_DIR / "results" / "training").glob(pattern))
    if not candidates:
        raise FileNotFoundError(
            f"No checkpoint found for batch_size={batch_size} matching "
            f"results/training/{pattern}"
        )
    if len(candidates) > 1:
        print(f"  WARNING: multiple checkpoints found for batch_size={batch_size}: "
              f"{[str(c) for c in candidates]}. Using the most recently modified.")
        candidates = sorted(candidates, key=lambda p: p.stat().st_mtime)
    return candidates[-1]


def cache_dir_for(checkpoint_path: Path, batch_size: int) -> Path:
    run_id = f"bs{batch_size}_{checkpoint_path.parent.name}"
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


def ensure_analyzed(batch_size: int, force: bool = False) -> Path:
    """Return the cache dir for this batch size, computing + caching only if needed."""
    checkpoint_path = find_checkpoint(batch_size)
    cdir = cache_dir_for(checkpoint_path, batch_size)

    if not force and has_cache(cdir):
        print(f"batch_size={batch_size}: using cached results at {cdir}")
        return cdir

    print(f"batch_size={batch_size}: no cached results at {cdir}, analyzing "
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

def load_batch_size_data(force: bool = False) -> pd.DataFrame:
    rows = []
    for bs in BATCH_SIZES:
        cdir = ensure_analyzed(bs, force=force)
        for target in TARGETS:
            roi_path = cdir / f"statistical_results_roi_{target}.csv"
            cons_path = cdir / f"consistency_brain_{target}.csv"
            if not (roi_path.exists() and cons_path.exists()):
                print(f"  WARNING: missing results for batch_size={bs}, target={target}; skipping.")
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
                    "roi":             roi,
                    "batch_size":      bs,
                    "delta":           r["delta"],
                    "p_value_fdr":     r["p_value_fdr"],
                    "significant_fdr": bool(r["significant"]),
                    "n_consistent":    n_consistent,
                    "n_subjects":      n_subjects,
                    "classification":  classification,
                })

    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "batch_size_roi_summary.csv"
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


def compute_shared_ylim(df: pd.DataFrame, pad_frac: float = 0.1) -> tuple:
    """Symmetric y-limit (centered on 0) covering every ROI/target/batch_size,
    so both output figures (vim, zi) are directly comparable by eye."""
    max_abs = df["delta"].abs().max()
    if not np.isfinite(max_abs) or max_abs == 0:
        return (-1.0, 1.0)
    padded = max_abs * (1 + pad_frac)
    return (-padded, padded)


def plot_target(df: pd.DataFrame, target: str, ylim: tuple):
    sub = df[df["target"] == target]
    rois = list(cpp.TARGET_ROIS)
    n_rois = len(rois)
    ncols = 4
    nrows = int(np.ceil(n_rois / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_data = sub[sub["roi"] == roi].set_index("batch_size").reindex(BATCH_SIZES)
        colors = [COLOR_MAP.get(c, "gray") for c in roi_data["classification"]]
        ax.bar([str(b) for b in BATCH_SIZES], roi_data["delta"].values, color=colors)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(ylim)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx // ncols == nrows - 1:
            ax.set_xlabel("batch size", fontsize=8)
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
    fig.suptitle(f"\u0394C by batch size, per ROI \u2014 {target.upper()}",
                fontsize=13, y=1.05)
    fig.tight_layout()

    out_path = OUT_DIR / f"roi_by_batch_size_{target}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# ENTRY POINT
# ================================================================================

def main(force: bool = False):
    df = load_batch_size_data(force=force)
    ylim = compute_shared_ylim(df)
    for target in TARGETS:
        plot_target(df, target, ylim=ylim)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-analyze every batch size even if cached "
                             "results already exist under results/batch_size/cache/.")
    args = parser.parse_args()
    main(force=args.force_recompute)