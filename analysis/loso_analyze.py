"""
================================================================================
LOSO Delta-C Analysis
================================================================================

For each of the 19 LOSO folds (results/loso_19_fold/fold_{subject}/best_model.pt),
loads that fold's model -- trained with the corresponding subject held out
entirely -- and computes decoder-projected ROI-space Delta_C (post - pre) for
that held-out subject only, for both VIM and ZI targets.

Per-subject "average of Delta_C VIM vs Delta_C ZI" is computed per ROI:
    delta_avg[subject, roi] = (delta_VIM[subject, roi] + delta_ZI[subject, roi]) / 2

Does NOT call compare_pre_post.main(); imports its functions directly
(load_model, compute_K, compute_roi_projection_weights, extract_C_diagonals,
project_to_roi, TARGET_ROIS), the same pattern as compare_batch_size.py and
compare_seed_effects.py.

ASSUMPTION (please confirm against compare_pre_post.py): extract_C_diagonals
(model, target) is assumed to compute diag(C) pre/post for every subject in
the full dataset (via cpp.load_all() internally), not just subjects the
passed-in model happened to be trained on -- this is what lets us run a LOSO
fold's model and pull out just the one held-out subject's row. If it actually
restricts to some other subset internally, this script's per-subject lookup
(by sid == held_out_subject) will raise a clear error for that fold rather
than silently producing wrong numbers -- but it's worth double-checking this
assumption against the actual function before trusting the output.

Output: results/loso_19_fold/
    loso_deltaC_per_subject_roi.csv   (subject x target x roi x delta, long format)
    loso_deltaC_averaged_roi.csv      (subject x roi x delta_avg, VIM/ZI averaged)
    loso_deltaC_summary.png           (per-ROI mean delta_avg across 19 held-out subjects,
                                        colored by sign-consistency: >=15/19 subjects agree)

Usage:
    python analysis/loso_analyze.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import compare_pre_post as cpp

RESULTS_DIR = ROOT_DIR / "results" / "loso_19_fold"
TARGETS = ["vim", "zi"]
CONSISTENCY_THRESHOLD = 15  # matches compare_pre_post's report_consistency_brain_space default


def find_fold_checkpoints() -> dict:
    """{held_out_subject_str: checkpoint_path} for every fold_* dir with a best_model_cls.pt"""
    ckpts = {}
    for fold_dir in sorted(RESULTS_DIR.glob("fold_*")):
        ckpt = fold_dir / "best_model_cls.pt"
        if ckpt.exists():
            held_out = fold_dir.name[len("fold_"):]
            ckpts[held_out] = ckpt
        else:
            print(f"  WARNING: no best_model_cls.pt in {fold_dir}, skipping.")
    return ckpts


def analyze_fold(checkpoint_path: Path, held_out_subject: str) -> pd.DataFrame:
    """Returns long-format rows [target, roi, delta] for just the held-out subject."""
    model = cpp.load_model(checkpoint_path)
    K, Lambda, W_bar_x = cpp.compute_K(model)
    roi_weights = cpp.compute_roi_projection_weights(W_bar_x)

    rows = []
    for target in TARGETS:
        sids, pre, post = cpp.extract_C_diagonals(model, target)
        sids = [str(s) for s in sids]
        if held_out_subject not in sids:
            raise ValueError(
                f"Held-out subject {held_out_subject} not found among extract_C_diagonals "
                f"results for target={target} (found: {sids}). See ASSUMPTION in module "
                f"docstring -- extract_C_diagonals may not cover all subjects for every model."
            )
        idx = sids.index(held_out_subject)
        pre_roi = cpp.project_to_roi(pre, roi_weights)
        post_roi = cpp.project_to_roi(post, roi_weights)
        delta_roi = post_roi[idx] - pre_roi[idx]   # (n_rois,)

        for roi, d in zip(cpp.TARGET_ROIS, delta_roi):
            rows.append({
                "subject": held_out_subject,
                "target": target,
                "roi": roi,
                "delta": d,
            })
    return pd.DataFrame(rows)


def main():
    ckpts = find_fold_checkpoints()
    if not ckpts:
        raise FileNotFoundError(f"No fold checkpoints found under {RESULTS_DIR}")
    print(f"Found {len(ckpts)} fold checkpoints.")

    all_rows = []
    for held_out, ckpt_path in ckpts.items():
        print(f"Analyzing fold held_out={held_out} ...")
        df_fold = analyze_fold(ckpt_path, held_out)
        all_rows.append(df_fold)

    df = pd.concat(all_rows, ignore_index=True)
    per_subject_path = RESULTS_DIR / "loso_deltaC_per_subject_roi.csv"
    df.to_csv(per_subject_path, index=False)
    print(f"\nSaved: {per_subject_path}")

    # --- Average Delta_C VIM vs Delta_C ZI, per subject per ROI ---
    pivot = df.pivot_table(index=["subject", "roi"], columns="target", values="delta").reset_index()
    missing_target = pivot[["vim", "zi"]].isna().any(axis=1)
    if missing_target.any():
        print(f"  WARNING: {missing_target.sum()} subject-roi rows missing one target; "
              f"dropping from averaged output.")
        pivot = pivot[~missing_target]
    pivot["delta_avg"] = (pivot["vim"] + pivot["zi"]) / 2

    avg_path = RESULTS_DIR / "loso_deltaC_averaged_roi.csv"
    pivot.to_csv(avg_path, index=False)
    print(f"Saved: {avg_path}")

    # --- Summary plot: mean delta_avg per ROI across all held-out subjects ---
    rois = list(cpp.TARGET_ROIS)
    means, sems, colors = [], [], []
    for roi in rois:
        vals = pivot[pivot["roi"] == roi]["delta_avg"].values
        means.append(vals.mean())
        sems.append(vals.std(ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0)
        n_pos = int((vals > 0).sum())
        n_neg = int((vals < 0).sum())
        n_consistent = max(n_pos, n_neg)
        colors.append("#1f77b4" if n_consistent >= CONSISTENCY_THRESHOLD else "#d62728")

    n_subjects = pivot["subject"].nunique()

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.bar(rois, means, yerr=sems, color=colors, capsize=3)
    ax.axhline(0, color="black", lw=0.6)
    ax.set_ylabel("Mean \u0394C, averaged VIM/ZI (decoder-projected)")
    ax.set_title(f"LOSO \u0394C per ROI, averaged over VIM/ZI, held-out subjects (n={n_subjects})")
    ax.tick_params(axis="x", rotation=60, labelsize=8)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color="#1f77b4",
                      label=f">={CONSISTENCY_THRESHOLD}/19 consistent direction"),
        plt.Rectangle((0, 0), 1, 1, color="#d62728", label="Below threshold"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", fontsize=8)
    fig.tight_layout()

    out_path = RESULTS_DIR / "loso_deltaC_summary.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()