"""
================================================================================
LOSO fold-specific ΔC (post - pre) by patient
================================================================================

For each completed LOSO fold in results/loso_19_fold/fold_{subject}/, loads
that fold's best_model_cls.pt (the model trained with `subject` held out
entirely) and evaluates it ONLY on `subject`'s own pre/post data -- every
value plotted here is a genuine held-out evaluation, never data the
checkpoint was trained on.

EXCLUDED FOLDS (see EXCLUDED_SUBJECTS below): sub-fuspd09 and sub-fuspd15
had reconstruction loss ~50-100x every other fold (228.9 and 205.9 vs a
0.5-3 range across the other 17), indicating those folds' models failed
to converge rather than genuinely fitting worse. Excluded by hand from all
downstream aggregate plots/stats -- NOT from the raw per-fold cache or the
loss table, so the convergence failure stays visible for diagnosis. Every
plot produced by this script states the exclusion explicitly in its title.

THREE OUTPUTS:
    1. Per-patient ΔC grids (one figure per target, unchanged design from
       the original version) -- raw, unaggregated, one bar per patient.
    2. Pooled per-ROI statistical significance, split into 4 groups by
       (target x treatment order) -- see "STATISTICAL TEST" below.
    3. Held-out loss-per-patient table (unfiltered -- this is how the two
       excluded folds were identified in the first place).

STATISTICAL TEST (pooled plot only):
    For each ROI, within each of 4 groups:
        VIM (1st tx):  target=vim, group_str=VIM_first
        VIM (2nd tx):  target=vim, group_str=ZI_first
        ZI  (1st tx):  target=zi,  group_str=ZI_first
        ZI  (2nd tx):  target=zi,  group_str=VIM_first
    a one-sample t-test (H0: mean delta = 0) is run across that group's
    patients. This is mathematically identical to a paired t-test on
    pre/post values, since `delta` is already the paired difference.
    BH-FDR correction is applied separately WITHIN each of the 4 groups
    across its 24 ROIs (4 independent corrections, not one pooled
    correction) -- deliberately not pooling 1st- and 2nd-treatment patients
    together for either the test or the correction, since averaging over
    treatment order could mask an order effect that's real but only visible
    within one arm of the crossover.

    Groups are tested and colored separately (not just visually split by a
    dotted line, as in the per-patient plots) precisely because "left half
    of the plot" and "statistically independent test" are different things
    -- the per-patient plots show a visual grouping, this test makes it a
    formal one.

COLORING (matches compare_batch_size.py / compare_seed_effects.py):
    green  = FDR-significant (BH-corrected, alpha=0.05, within-group)
    blue   = not FDR-significant, but consistent direction across
             >= CONSISTENCY_FRACTION of that group's patients (a
             distribution-free check, since one-sample t-tests get shaky
             at the small per-group N here -- roughly 8-9 after exclusion)
    red    = neither
    Green explicitly takes priority over blue: a ROI/group that is BOTH
    FDR-significant AND direction-consistent is colored green, never blue
    -- enforced by evaluating the FDR condition first and short-circuiting
    on it (see classify() below).

CAUTION -- per-fold N=1, pooled N=17 (after exclusion): each individual
fold only ever tells you ΔC for its one held-out patient; the statistical
test is run on the POOLED table across all folds (each fold contributing
exactly one patient's ΔC, computed from a model that never saw that
patient), not per-fold. Consistent with the paired t-test + BH-FDR
approach used everywhere else in this project.

SHARED DEPENDENCY: model loading, K computation, and decoder-based ROI
projection are imported from analysis.analysis_helper_functions (the same
functions used by compare_pre_post.py, compare_batch_size.py, and
compare_seed_effects.py). evaluate_fold() below does its own C extraction
rather than using analysis_helper_functions.extract_all_C, because it
calls the model with a different signature --
model(x, label, kl_g0_weight=..., kl_u_weight=..., apply_free_bits=...)
via BRICKDataset items, returning a dict with "C" (full MxM matrix) and
"losses" -> {"loss_recon": ..., "loss_cls": ...} -- mirroring train.py's
run_epoch() usage rather than the plain model(x) call extract_all_C
assumes.

Usage:
    python analysis/loso_analyze.py
    python analysis/loso_analyze.py --force-recompute
"""

import sys
import math
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from scipy import stats as sstats

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from preprocessing.load_preprocessed_data import TARGET_ROIS
from analysis.analysis_helper_functions import (
    load_model, compute_K, compute_roi_projection_weights, project_to_roi,
)
from training.dataset import BRICKDataset
from training.train import DATA_DIR

TARGETS = ["vim", "zi"]
TARGET_FIRST_LABEL = {"vim": "VIM_first", "zi": "ZI_first"}
TARGET_COLOR = {"vim": "#4C72B0", "zi": "#DD8452"}   # blue / orange

# Folds excluded by hand: reconstruction loss ~50-100x every other fold
# (228.9, 205.9 vs a 1.5-9.7 range across the other 17) -- see loso_loss_by_patient.csv.
# Applied to every aggregate plot/stat below; NOT applied to the raw per-fold
# cache or the loss table, so the convergence failure stays visible.
EXCLUDED_SUBJECTS = {"sub-fuspd09", "sub-fuspd15", "sub-fuspd19"}

ALPHA = 0.05
# Distribution-free consistency check, scaled to whatever N a given group has.
# 15/19 matches the fixed threshold used elsewhere in this project (compare_
# batch_size.py, compare_seed_effects.py), which was calibrated for N=19;
# here group sizes are much smaller (~8-9 after exclusion), so the same
# *proportion* is applied via ceil() rather than reusing the fixed count 15.
CONSISTENCY_FRACTION = 15 / 19

COLOR_MAP = {
    "fdr_significant":       "#2ca02c",   # green
    "consistency_threshold": "#1f77b4",   # blue
    "neither":                "#d62728",  # red
}

LOSO_DIR  = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2"
OUT_DIR   = LOSO_DIR / "results" / "loso_19_fold_beta_0.2"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TREATMENT_GROUPS = [
    # (display_label, target, group_str_filter)
    ("VIM (1st tx)", "vim", "VIM_first"),
    ("VIM (2nd tx)", "vim", "ZI_first"),
    ("ZI (1st tx)",  "zi",  "ZI_first"),
    ("ZI (2nd tx)",  "zi",  "VIM_first"),
]


# ================================================================================
# LOCATE FOLDS
# ================================================================================
def find_folds() -> dict:
    """
    {subject_id: checkpoint_path} for every results/loso_19_fold/fold_{subject_id}/
    that has a best_model_cls.pt. Subject IDs come from the folder name
    (fold_{subject_id}), matching loso_study.py's run_name convention.
    """
    folds = {}
    if not LOSO_DIR.exists():
        return folds
    for d in sorted(LOSO_DIR.glob("fold_*")):
        if not d.is_dir():
            continue
        ckpt = d / "best_model_cls.pt"
        if ckpt.exists():
            subject_id = d.name[len("fold_"):]
            folds[subject_id] = ckpt
        else:
            print(f"  Skipping {d.name}: no best_model_cls.pt yet.")
    return folds


# ================================================================================
# LOAD ALL PATIENT DATA (via BRICKDataset directly -- see module docstring)
# ================================================================================
CONDITION_TO_SESSION = {"mpre": "pre", "mpost": "post"}

def load_all_items() -> list:
    """
    One dict per (subject, target, session) triple, x already z-scored
    exactly as BRICKDataset produces it for training.
    """
    ds = BRICKDataset(DATA_DIR)
    items = []
    for i in range(len(ds)):
        item = ds[i]
        items.append({
            "subject_id":      item["subject_id"],
            "target":          item["target"],
            "session":         CONDITION_TO_SESSION[item["condition_str"]],
            "group_str":       item["group_str"],
            "x":               item["x"],
            # NOTE: do NOT cast to int here -- BRICKDataset returns this as a
            # tensor, and model(x, label, ...) requires label.unsqueeze(0) to
            # work inside brick.py's _compute_losses. Casting to a plain int
            # (as an earlier version of this script did) caused
            # AttributeError: 'int' object has no attribute 'unsqueeze'.
            "lifus_condition": item["lifus_condition"],
        })
    return items


# ================================================================================
# EVALUATE ONE FOLD ON ITS OWN HELD-OUT PATIENT
# ================================================================================
def evaluate_fold(subject_id: str, checkpoint_path: Path, all_items: list) -> list:
    """
    Load this fold's checkpoint and evaluate it ONLY on subject_id's own
    pre/post data, computing ΔC = post - pre per ROI for each target the
    subject has. Returns a list of row-dicts: one per (target, roi).
    Each row carries both the decoder-projected pre/post values and their
    delta, so downstream analyses can reconstruct either the paired
    difference or the raw per-session trajectory without re-running the
    model.
    """
    my_items = [it for it in all_items if it["subject_id"] == subject_id]
    if not my_items:
        print(f"  WARNING: no data found for subject_id={subject_id} -- check "
              f"the ID format matches BRICKDataset exactly (e.g. 'sub-fuspd13').")
        return []

    model = load_model(checkpoint_path)
    K, Lambda, W_bar_x = compute_K(model)
    roi_weights = compute_roi_projection_weights(W_bar_x)

    rows = []
    with torch.no_grad():
        for target in TARGETS:
            pre_item  = next((it for it in my_items
                               if it["target"] == target and it["session"] == "pre"), None)
            post_item = next((it for it in my_items
                               if it["target"] == target and it["session"] == "post"), None)
            if pre_item is None or post_item is None:
                continue

            if pre_item["group_str"] != post_item["group_str"]:
                print(f"  WARNING: {subject_id}/{target} has mismatched group_str "
                      f"between pre ({pre_item['group_str']}) and post "
                      f"({post_item['group_str']}) items -- using the pre value.")
            group_str = pre_item["group_str"]

            roi_c, losses = {}, {}
            for session, item in [("pre", pre_item), ("post", post_item)]:
                out = model(item["x"], item["lifus_condition"],
                            kl_g0_weight=1.0, kl_u_weight=1.0, apply_free_bits=False)

                C = out["C"]
                C_diag = torch.diagonal(C)
                if C_diag.is_complex():
                    C_diag = C_diag.real
                roi_c[session] = project_to_roi(C_diag.unsqueeze(0), roi_weights)[0]

                losses[session] = {
                    "loss_recon": out["losses"]["loss_recon"].item(),
                    "loss_cls":   out["losses"]["loss_cls"].item(),
                }

            delta = roi_c["post"] - roi_c["pre"]

            for roi_idx, roi_name in enumerate(TARGET_ROIS):
                rows.append({
                    "subject_id":       subject_id,
                    "target":           target,
                    "group_str":        group_str,
                    "roi":              roi_name,
                    "roi_c_pre":        float(roi_c["pre"][roi_idx]),
                    "roi_c_post":       float(roi_c["post"][roi_idx]),
                    "delta":            float(delta[roi_idx]),
                    "loss_recon_pre":   losses["pre"]["loss_recon"],
                    "loss_recon_post":  losses["post"]["loss_recon"],
                    "loss_cls_pre":     losses["pre"]["loss_cls"],
                    "loss_cls_post":    losses["post"]["loss_cls"],
                })
    return rows


def ensure_evaluated(subject_id: str, checkpoint_path: Path, all_items: list,
                      force: bool = False) -> Path:
    cache_path = CACHE_DIR / f"{subject_id}.csv"
    if not force and cache_path.exists():
        print(f"{subject_id}: using cached results at {cache_path}")
        return cache_path

    print(f"{subject_id}: no cache, evaluating {checkpoint_path} on held-out data...")
    rows = evaluate_fold(subject_id, checkpoint_path, all_items)
    if not rows:
        return cache_path
    pd.DataFrame(rows).to_csv(cache_path, index=False)
    print(f"  Cached to {cache_path}")
    return cache_path


# ================================================================================
# AGGREGATE
# ================================================================================
def load_loso_data(force: bool = False):
    """Returns (df_filtered, df_raw, excluded_present) --
    df_raw: every completed fold, unfiltered (used for the loss table, so
            the convergence failure that motivated the exclusion stays visible).
    df_filtered: EXCLUDED_SUBJECTS removed (used for every ΔC plot/stat).
    excluded_present: EXCLUDED_SUBJECTS that were actually found among the
            completed folds (for explicit labeling on plots)."""
    folds = find_folds()
    if not folds:
        raise FileNotFoundError(
            f"No completed folds with best_model_cls.pt found under {LOSO_DIR}"
        )

    print(f"Found {len(folds)} completed fold(s): {sorted(folds.keys())}")
    all_items = load_all_items()

    frames = []
    for subject_id, checkpoint_path in folds.items():
        cache_path = ensure_evaluated(subject_id, checkpoint_path, all_items, force=force)
        if cache_path.exists():
            frames.append(pd.read_csv(cache_path))

    if not frames:
        raise RuntimeError("No fold produced any evaluated rows -- check the "
                            "subject_id-format WARNING messages above.")

    df_raw = pd.concat(frames, ignore_index=True)
    raw_csv = OUT_DIR / "loso_delta_summary.csv"
    df_raw.to_csv(raw_csv, index=False)
    print(f"\nSaved aggregated summary (all completed folds): {raw_csv}")

    excluded_present = sorted(EXCLUDED_SUBJECTS & set(df_raw["subject_id"]))
    if excluded_present:
        print(f"Excluding {excluded_present} from downstream ΔC plots/stats "
              f"(see EXCLUDED_SUBJECTS in module header) -- raw data above is unaffected.")

    df_filtered = df_raw[~df_raw["subject_id"].isin(EXCLUDED_SUBJECTS)].reset_index(drop=True)
    filtered_csv = OUT_DIR / "loso_delta_summary_filtered.csv"
    df_filtered.to_csv(filtered_csv, index=False)
    print(f"Saved filtered summary (n={df_filtered['subject_id'].nunique()} patients): {filtered_csv}")

    return df_filtered, df_raw, excluded_present


# ================================================================================
# PATIENT ORDERING (crossover-order grouping)
# ================================================================================
def order_patients(df: pd.DataFrame, target: str) -> tuple:
    """
    Returns (first_group, second_group): sorted lists of subject IDs.
    first_group = patients for whom `target` was their FIRST treatment
    (group_str == TARGET_FIRST_LABEL[target]).
    second_group = everyone else with data for this target.
    """
    sub = df[df["target"] == target]
    first_label = TARGET_FIRST_LABEL[target]
    subj_group = sub.drop_duplicates("subject_id").set_index("subject_id")["group_str"]
    first_group  = sorted(s for s in subj_group.index if subj_group[s] == first_label)
    second_group = sorted(s for s in subj_group.index if subj_group[s] != first_label)
    return first_group, second_group


# ================================================================================
# SHARED Y-LIMITS
# ================================================================================
def compute_shared_ylim(values, pad_frac: float = 0.1) -> tuple:
    """Symmetric y-limit (centered on 0) covering the given array of values."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) == 0:
        return (-1.0, 1.0)
    max_abs = np.max(np.abs(values))
    if max_abs == 0:
        return (-1.0, 1.0)
    padded = max_abs * (1 + pad_frac)
    return (-padded, padded)


def exclusion_note(excluded_subjects) -> str:
    if not excluded_subjects:
        return ""
    return f"Excluded (did not converge): {', '.join(sorted(excluded_subjects))}"


# ================================================================================
# PLOT 1 & 2 — per-patient ΔC grid, one figure per target (unchanged design)
# ================================================================================
def plot_delta_grid(df: pd.DataFrame, target: str, ylim: tuple, excluded_subjects):
    sub = df[df["target"] == target]
    if sub.empty:
        print(f"  No data for target={target}, skipping plot.")
        return

    rois = list(TARGET_ROIS)
    first_group, second_group = order_patients(df, target)
    patients = first_group + second_group
    split_idx = len(first_group)

    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))
    color = TARGET_COLOR[target]

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_df = sub[sub["roi"] == roi].set_index("subject_id").reindex(patients)
        x = np.arange(len(patients))
        ax.bar(x, roi_df["delta"].values, color=color)
        ax.axhline(0, color="black", lw=0.6)
        if 0 < split_idx < len(patients):
            ax.axvline(split_idx - 0.5, color="black", lw=1.0, linestyle=":")
        ax.set_ylim(ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(patients, rotation=90, fontsize=6)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("\u0394C = post \u2212 pre (raw)", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    first_label  = TARGET_FIRST_LABEL[target]
    other_label  = [v for v in TARGET_FIRST_LABEL.values() if v != first_label][0]
    title = (
        f"LOSO held-out \u0394C by patient, per ROI \u2014 {target.upper()} treatment\n"
        f"left of dotted line: {first_label} (n={len(first_group)})   |   "
        f"right: {other_label} (n={len(second_group)})"
    )
    note = exclusion_note(excluded_subjects)
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=12, y=1.06 if note else 1.05)
    fig.tight_layout()

    out_path = OUT_DIR / f"loso_delta_by_patient_{target}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# STATISTICS — per-ROI, per-treatment-group, BH-FDR corrected
# ================================================================================
def bh_fdr(pvals: np.ndarray, alpha: float = ALPHA) -> np.ndarray:
    """Benjamini-Hochberg FDR-adjusted p-values. NaNs pass through as NaN."""
    pvals = np.asarray(pvals, dtype=float)
    valid = np.isfinite(pvals)
    adj = np.full_like(pvals, np.nan)
    if valid.sum() == 0:
        return adj

    p_valid = pvals[valid]
    n = len(p_valid)
    order = np.argsort(p_valid)
    ranked = p_valid[order]
    raw_adj = ranked * n / (np.arange(1, n + 1))
    # enforce monotonicity from the largest p-value downward
    raw_adj = np.minimum.accumulate(raw_adj[::-1])[::-1]
    raw_adj = np.clip(raw_adj, 0, 1)

    adj_valid = np.empty(n)
    adj_valid[order] = raw_adj
    adj[valid] = adj_valid
    return adj


def classify(significant_fdr: bool, consistent_direction: bool) -> str:
    """Green (FDR-significant) explicitly takes priority over blue
    (direction-consistent): a ROI/group that is both is classified green,
    never blue. Enforced by checking the FDR condition first."""
    if significant_fdr:
        return "fdr_significant"
    elif consistent_direction:
        return "consistency_threshold"
    else:
        return "neither"


def compute_pooled_stats(df: pd.DataFrame, alpha: float = ALPHA) -> pd.DataFrame:
    """One-sample t-test per ROI (H0: mean delta = 0), run separately within
    each of the 4 TREATMENT_GROUPS, with BH-FDR correction applied
    independently within each group across its 24 ROIs."""
    rois = list(TARGET_ROIS)
    all_rows = []

    for label, target, group_str in TREATMENT_GROUPS:
        sub = df[(df["target"] == target) & (df["group_str"] == group_str)]
        n_subjects = sub["subject_id"].nunique()

        roi_stats = []
        for roi in rois:
            vals = sub[sub["roi"] == roi]["delta"].values
            n = len(vals)
            if n >= 2:
                mean = float(np.mean(vals))
                sem = float(np.std(vals, ddof=1) / np.sqrt(n))
                _, p = sstats.ttest_1samp(vals, 0.0)
            elif n == 1:
                mean = float(vals[0])
                sem = np.nan
                p = np.nan
            else:
                mean, sem, p = np.nan, np.nan, np.nan

            n_pos = int((vals > 0).sum())
            n_neg = int((vals < 0).sum())
            n_consistent = max(n_pos, n_neg)
            consistency_threshold = math.ceil(CONSISTENCY_FRACTION * n) if n > 0 else 0
            consistent_direction = n > 0 and n_consistent >= consistency_threshold

            roi_stats.append({
                "group_label": label, "target": target, "group_str": group_str,
                "roi": roi, "n_subjects_group": n_subjects, "n_roi": n,
                "mean_delta": mean, "sem": sem, "p_value": p,
                "n_consistent": n_consistent, "consistency_threshold": consistency_threshold,
                "consistent_direction": consistent_direction,
            })

        # BH-FDR within this group, across its 24 ROIs
        pvals = np.array([r["p_value"] for r in roi_stats])
        p_fdr = bh_fdr(pvals, alpha=alpha)
        for r, pf in zip(roi_stats, p_fdr):
            r["p_value_fdr"] = pf
            r["significant_fdr"] = bool(np.isfinite(pf) and pf <= alpha)
            r["classification"] = classify(r["significant_fdr"], r["consistent_direction"])
            all_rows.append(r)

    return pd.DataFrame(all_rows)


# ================================================================================
# PLOT 3 — pooled significance, 4 bars per ROI
# ================================================================================
def plot_pooled_grid(stats_df: pd.DataFrame, ylim: tuple, excluded_subjects):
    rois = list(TARGET_ROIS)
    labels_order = [g[0] for g in TREATMENT_GROUPS]

    ncols = 4
    nrows = int(np.ceil(len(rois) / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_stats = stats_df[stats_df["roi"] == roi].set_index("group_label").reindex(labels_order)
        colors = [COLOR_MAP.get(c, "gray") for c in roi_stats["classification"]]
        x = np.arange(len(labels_order))
        ax.bar(x, roi_stats["mean_delta"].values, yerr=roi_stats["sem"].values,
               color=colors, capsize=3)
        ax.axhline(0, color="black", lw=0.6)
        ax.set_ylim(ylim)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_order, rotation=30, fontsize=6.5, ha="right")
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Mean \u0394C \u00b1 SEM", fontsize=8)

    for idx in range(len(rois), nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    n_by_group = {g[0]: stats_df[stats_df["group_label"] == g[0]]["n_subjects_group"].iloc[0]
                  for g in TREATMENT_GROUPS}
    n_str = "   |   ".join(f"{lbl}: n={n_by_group[lbl]}" for lbl in labels_order)

    legend_handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["fdr_significant"],
                      label="FDR significant (within-group, \u03b1=0.05)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["consistency_threshold"],
                      label=f"Not FDR-sig., but \u2265{CONSISTENCY_FRACTION:.0%} consistent direction "
                            f"(green takes priority if both apply)"),
        plt.Rectangle((0, 0), 1, 1, color=COLOR_MAP["neither"], label="Neither"),
    ]
    fig.legend(handles=legend_handles, loc="upper center", ncol=1, fontsize=8.5,
               bbox_to_anchor=(0.5, 1.03))

    title = f"LOSO pooled \u0394C significance by ROI, per treatment group\n{n_str}"
    note = exclusion_note(excluded_subjects)
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=12, y=1.10 if note else 1.08)
    fig.tight_layout()

    out_path = OUT_DIR / "loso_pooled_significance_by_roi.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# PLOT 4 — held-out C timeline grid, one panel per ROI, EXCLUDED_SUBJECTS removed
# ================================================================================

C_STAGES = ["pre_tx1", "post_tx1", "pre_tx2", "post_tx2"]
C_STAGE_LABELS_SHORT = ["Pre-1", "Post-1", "Pre-2", "Post-2"]
C_GROUP_COLORS = {"vim_first": "#d62728", "zi_first": "#1f77b4"}

# From the study's Methods (Visit 2): both sessions occur the same day,
# separated by a mean of 3.8h (SD 0.2). Annotation only -- not used in
# any numerical computation.
INTER_SESSION_GAP_HOURS_MEAN = 3.8
INTER_SESSION_GAP_HOURS_SD = 0.2


def build_roi_timeline(df: pd.DataFrame) -> dict:
    """
    Reshape the (already exclusion-filtered) LOSO dataframe from
    long (subject, target, roi) rows into per-subject 4-stage trajectories,
    using each subject's group_str to determine which target was tx1 vs tx2.

    Returns:
        {
          subject_id: {
            "order": "vim_first" | "zi_first",
            "pre_tx1": (N_ROIS,), "post_tx1": (N_ROIS,),
            "pre_tx2": (N_ROIS,), "post_tx2": (N_ROIS,),
          },
          ...
        }
    Subjects missing either target, or with inconsistent group_str across
    targets, are skipped with a warning.
    """
    rois = list(TARGET_ROIS)
    out = {}

    for sid, sub in df.groupby("subject_id"):
        group_strs = sub["group_str"].unique()
        if len(group_strs) > 1:
            print(f"  WARNING: {sid} has inconsistent group_str across targets "
                  f"{list(group_strs)} -- skipping.")
            continue
        group_str = group_strs[0]
        if group_str not in TARGET_FIRST_LABEL.values():
            print(f"  WARNING: {sid} has unrecognized group_str {group_str!r} -- skipping.")
            continue

        first = "vim" if group_str == TARGET_FIRST_LABEL["vim"] else "zi"
        second = "zi" if first == "vim" else "vim"
        order = f"{first}_first"

        sub_first = sub[sub["target"] == first].set_index("roi")
        sub_second = sub[sub["target"] == second].set_index("roi")
        if len(sub_first) != len(rois) or len(sub_second) != len(rois):
            print(f"  WARNING: {sid} missing ROI rows for one or both targets -- skipping.")
            continue

        out[sid] = {
            "order":    order,
            "pre_tx1":  sub_first.loc[rois, "roi_c_pre"].values.astype(float),
            "post_tx1": sub_first.loc[rois, "roi_c_post"].values.astype(float),
            "pre_tx2":  sub_second.loc[rois, "roi_c_pre"].values.astype(float),
            "post_tx2": sub_second.loc[rois, "roi_c_post"].values.astype(float),
        }

    return out


def plot_roi_timeline_grid(C_roi_all: dict, roi_names, out_path: Path, excluded_subjects):
    """
    24-panel grid, one panel per ROI: mean +/- SEM decoder-projected C at
    pre-tx1/post-tx1/pre-tx2/post-tx2, one line per treatment order.
    Each panel's y-axis is sized to its own data (mean +/- SEM across all
    4 stages/both groups), but every panel uses the same axis SPAN, so
    spread is visually comparable across ROIs without forcing 0 into view.
    """
    rois = list(roi_names)
    n_rois = len(rois)
    ncols = 4
    nrows = int(np.ceil(n_rois / ncols))

    group_stage_vals = {order: {s: [] for s in C_STAGES} for order in C_GROUP_COLORS}
    group_n = {order: 0 for order in C_GROUP_COLORS}
    for sid, d in C_roi_all.items():
        order = d["order"]
        if order not in group_stage_vals:
            continue
        for s in C_STAGES:
            group_stage_vals[order][s].append(d[s])
        group_n[order] += 1

    for order in group_stage_vals:
        for s in C_STAGES:
            group_stage_vals[order][s] = (
                np.stack(group_stage_vals[order][s]) if group_stage_vals[order][s] else None
            )

    per_roi_data = []
    for idx in range(n_rois):
        roi_data = {}
        lo, hi = np.inf, -np.inf
        for order in C_GROUP_COLORS:
            n = group_n[order]
            if n == 0:
                continue
            means, sems = [], []
            for s in C_STAGES:
                vals = group_stage_vals[order][s]
                col = vals[:, idx]
                m = col.mean()
                e = col.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                means.append(m)
                sems.append(e)
                lo = min(lo, m - e)
                hi = max(hi, m + e)
            roi_data[order] = (means, sems, n)
        roi_data["_range"] = (lo, hi)
        per_roi_data.append(roi_data)

    spans = [hi - lo for (lo, hi) in (d["_range"] for d in per_roi_data) if np.isfinite(hi - lo)]
    max_span = (max(spans) if spans else 1.0) * 1.15

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_data = per_roi_data[idx]

        for order, color in C_GROUP_COLORS.items():
            if order not in roi_data:
                continue
            means, sems, n = roi_data[order]
            ax.errorbar(range(4), means, yerr=sems, color=color, marker="o",
                        capsize=3, lw=1.5, markersize=4,
                        label=f"{order.split('_')[0].upper()}-first (n={n})")

        lo, hi = roi_data["_range"]
        if np.isfinite(lo) and np.isfinite(hi):
            center = (lo + hi) / 2
            ax.set_ylim(center - max_span / 2, center + max_span / 2)

        ax.axhline(0, color="black", lw=0.5, zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(C_STAGE_LABELS_SHORT, fontsize=7)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("Mean held-out C (decoder-projected)", fontsize=8)

    for idx in range(n_rois, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))

    title = (
        "LOSO held-out C by ROI across treatment sessions, per treatment order\n"
        f"(post-TUS-1 \u2192 pre-TUS-2 gap: ~{INTER_SESSION_GAP_HOURS_MEAN:.1f}h, "
        f"SD {INTER_SESSION_GAP_HOURS_SD:.1f})"
    )
    note = exclusion_note(excluded_subjects)
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=12, y=1.08 if note else 1.06)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


   # ================================================================================
# PLOT 5 — paired cumulative ΔC from baseline, one panel per ROI
# ================================================================================

C_DELTA_STAGES = ["baseline", "after_tx1", "after_wait", "after_tx2"]
C_DELTA_STAGE_LABELS = ["Baseline\n(pre-TUS 1)", "After TUS 1", "After wait", "After TUS 2"]


def build_roi_cumulative_delta(C_roi_all: dict) -> dict:
    """
    For each subject, compute cumulative ΔC relative to their own pre-tx1
    baseline at 3 later checkpoints, plus an explicit 0 at baseline. Since
    every stage is (stage_value - that subject's own pre_tx1), each
    subject's individual baseline level cancels out -- same logic as
    `delta` (post - pre) elsewhere in this script -- so the resulting SEM
    reflects within-subject change, not between-subject baseline spread.

    Returns:
        {
          subject_id: {
            "order": "vim_first" | "zi_first",
            "baseline":   (N_ROIS,) zeros,
            "after_tx1":  (N_ROIS,),   # post_tx1 - pre_tx1
            "after_wait": (N_ROIS,),   # pre_tx2  - pre_tx1
            "after_tx2":  (N_ROIS,),   # post_tx2 - pre_tx1
          },
          ...
        }
    """
    out = {}
    for sid, d in C_roi_all.items():
        baseline = d["pre_tx1"]
        out[sid] = {
            "order":      d["order"],
            "baseline":   np.zeros_like(baseline),
            "after_tx1":  d["post_tx1"] - baseline,
            "after_wait": d["pre_tx2"]  - baseline,
            "after_tx2":  d["post_tx2"] - baseline,
        }
    return out


def plot_roi_cumulative_delta_grid(C_delta_all: dict, roi_names, out_path: Path, excluded_subjects):
    """
    24-panel grid, one panel per ROI: mean +/- SEM cumulative ΔC (relative
    to each subject's own pre-tx1 baseline) at baseline/after-tx1/after-
    wait/after-tx2, one line per treatment order. Paired at the subject
    level, so SEM reflects within-subject variability only -- comparable in
    scale to loso_pooled_significance_by_roi.png, unlike the raw-value
    timeline. Y-axis: same span-matching approach as plot_roi_timeline_grid,
    but symmetric around 0 since these are signed deltas from a 0 baseline.
    """
    rois = list(roi_names)
    n_rois = len(rois)
    ncols = 4
    nrows = int(np.ceil(n_rois / ncols))

    group_stage_vals = {order: {s: [] for s in C_DELTA_STAGES} for order in C_GROUP_COLORS}
    group_n = {order: 0 for order in C_GROUP_COLORS}
    for sid, d in C_delta_all.items():
        order = d["order"]
        if order not in group_stage_vals:
            continue
        for s in C_DELTA_STAGES:
            group_stage_vals[order][s].append(d[s])
        group_n[order] += 1

    for order in group_stage_vals:
        for s in C_DELTA_STAGES:
            group_stage_vals[order][s] = (
                np.stack(group_stage_vals[order][s]) if group_stage_vals[order][s] else None
            )

    per_roi_data = []
    for idx in range(n_rois):
        roi_data = {}
        max_abs = 0.0
        for order in C_GROUP_COLORS:
            n = group_n[order]
            if n == 0:
                continue
            means, sems = [], []
            for s in C_DELTA_STAGES:
                vals = group_stage_vals[order][s]
                col = vals[:, idx]
                m = col.mean()
                e = col.std(ddof=1) / np.sqrt(n) if n > 1 else 0.0
                means.append(m)
                sems.append(e)
                max_abs = max(max_abs, abs(m) + e)
            roi_data[order] = (means, sems, n)
        roi_data["_max_abs"] = max_abs
        per_roi_data.append(roi_data)

    spans = [d["_max_abs"] for d in per_roi_data if np.isfinite(d["_max_abs"]) and d["_max_abs"] > 0]
    max_span = (max(spans) if spans else 1.0) * 2 * 1.15  # symmetric span, padded

    fig, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 3 * nrows), squeeze=False)

    for idx, roi in enumerate(rois):
        ax = axes[idx // ncols][idx % ncols]
        roi_data = per_roi_data[idx]

        for order, color in C_GROUP_COLORS.items():
            if order not in roi_data:
                continue
            means, sems, n = roi_data[order]
            ax.errorbar(range(4), means, yerr=sems, color=color, marker="o",
                        capsize=3, lw=1.5, markersize=4,
                        label=f"{order.split('_')[0].upper()}-first (n={n})")

        ax.set_ylim(-max_span / 2, max_span / 2)
        ax.axhline(0, color="black", lw=0.5, zorder=0)
        ax.set_xticks(range(4))
        ax.set_xticklabels(C_DELTA_STAGE_LABELS, fontsize=6.5)
        ax.set_title(roi, fontsize=9)
        ax.tick_params(labelsize=7)
        if idx % ncols == 0:
            ax.set_ylabel("\u0394C from baseline (decoder-projected)", fontsize=8)

    for idx in range(n_rois, nrows * ncols):
        axes[idx // ncols][idx % ncols].axis("off")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=9,
               bbox_to_anchor=(0.5, 1.03))

    title = (
        "LOSO held-out cumulative \u0394C from baseline, by ROI, per treatment order\n"
        f"(post-TUS-1 \u2192 pre-TUS-2 gap: ~{INTER_SESSION_GAP_HOURS_MEAN:.1f}h, "
        f"SD {INTER_SESSION_GAP_HOURS_SD:.1f})"
    )
    note = exclusion_note(excluded_subjects)
    if note:
        title += f"\n{note}"
    fig.suptitle(title, fontsize=12, y=1.08 if note else 1.06)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# LOSS TABLE — one row per patient (not split by target, unfiltered)
# ================================================================================
def build_loss_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient, averaging recon/cls loss across every available
    (target, session) evaluation for that patient's fold checkpoint --
    VIM-pre, VIM-post, ZI-pre, ZI-post, wherever present. The checkpoint is
    per-patient (not per-target), so VIM- and ZI-session losses for the same
    patient are expected to be close; this collapses them into a single
    per-patient number rather than reporting near-duplicate rows per target.

    Deliberately uses df_raw (unfiltered, all completed folds) rather than
    the exclusion-filtered df -- this table is exactly how the two excluded
    folds were identified, so it needs to keep showing them.
    """
    loss_cols = ["loss_recon_pre", "loss_recon_post", "loss_cls_pre", "loss_cls_post"]
    per_target = df_raw.drop_duplicates(subset=["subject_id", "target"])[
        ["subject_id", "target"] + loss_cols
    ].copy()

    per_target["loss_recon_mean"] = per_target[["loss_recon_pre", "loss_recon_post"]].mean(axis=1)
    per_target["loss_cls_mean"]   = per_target[["loss_cls_pre",   "loss_cls_post"]].mean(axis=1)

    table = (
        per_target.groupby("subject_id")
        .agg(loss_recon=("loss_recon_mean", "mean"),
             loss_cls=("loss_cls_mean", "mean"))
        .reset_index()
        .sort_values("subject_id")
        .reset_index(drop=True)
    )
    table["excluded"] = table["subject_id"].isin(EXCLUDED_SUBJECTS)
    return table


def print_loss_table(table: pd.DataFrame):
    print(f"\nHeld-out loss per patient (n={len(table)}, averaged across "
          f"VIM/ZI and pre/post; 'excluded' flags folds dropped from ΔC plots/stats):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    # --- Mean summary rows, both with and without the excluded outlier folds ---
    # Shown separately rather than one blended number: sub-fuspd09/15 are
    # 50-100x every other fold's loss, so a single mean across all 19 would
    # be dominated by those two and wouldn't reflect "typical" accuracy.
    included = table[~table["excluded"]]
    summary_rows = pd.DataFrame([
        {
            "subject_id": f"MEAN (n={len(included)}, excludes outlier folds)",
            "loss_recon": included["loss_recon"].mean(),
            "loss_cls":   included["loss_cls"].mean(),
            "excluded":   "",
        },
        {
            "subject_id": f"MEAN (n={len(table)}, all patients)",
            "loss_recon": table["loss_recon"].mean(),
            "loss_cls":   table["loss_cls"].mean(),
            "excluded":   "",
        },
    ])

    print()
    print(summary_rows.to_string(
        index=False,
        formatters={
            "loss_recon": lambda v: f"{v:.4f}",
            "loss_cls":   lambda v: f"{v:.4f}",
        },
    ))
    print(
        f"\n(Recon loss and cls loss aren't on comparable scales -- recon is a "
        f"reconstruction MSE-derived term, cls is a binary classification loss -- "
        f"so 'more accurate' here means comparing each metric across patients, "
        f"not recon vs. cls directly against each other.)"
    )

    out_csv = OUT_DIR / "loso_loss_by_patient.csv"
    table_with_summary = pd.concat([table, summary_rows], ignore_index=True)
    table_with_summary.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


# ================================================================================
# ENTRY POINT
# ================================================================================
def main(force: bool = False):
    df, df_raw, excluded_present = load_loso_data(force=force)

    # --- Plots 1 & 2: per-patient ΔC grids (exclusion-filtered) ---
    ylim = compute_shared_ylim(df["delta"].values)
    for target in TARGETS:
        plot_delta_grid(df, target, ylim=ylim, excluded_subjects=excluded_present)

    # --- Plot 3: pooled significance by treatment group ---
    stats_df = compute_pooled_stats(df)
    stats_csv = OUT_DIR / "loso_pooled_stats.csv"
    stats_df.to_csv(stats_csv, index=False)
    print(f"Saved {stats_csv}")

    n_sig = int(stats_df["significant_fdr"].sum())
    n_consistent_only = int(((stats_df["classification"] == "consistency_threshold")).sum())
    print(f"\nPooled stats summary: {n_sig}/{len(stats_df)} ROI-group cells FDR-significant, "
          f"{n_consistent_only} more meeting the consistency threshold only.")

    pooled_ylim = compute_shared_ylim(stats_df["mean_delta"].values)
    plot_pooled_grid(stats_df, ylim=pooled_ylim, excluded_subjects=excluded_present)

    # --- Plot 4: held-out C timeline grid (exclusion-filtered) ---
    C_roi_all = build_roi_timeline(df)
    print(f"\nBuilt timeline for {len(C_roi_all)} subjects (excluded: {excluded_present}).")
    plot_roi_timeline_grid(C_roi_all, TARGET_ROIS,
                           OUT_DIR / "loso_C_timeline_by_roi.png",
                           excluded_subjects=excluded_present)

    # --- Plot 5: paired cumulative delta from baseline ---
    C_delta_all = build_roi_cumulative_delta(C_roi_all)
    plot_roi_cumulative_delta_grid(C_delta_all, TARGET_ROIS,
                                   OUT_DIR / "loso_C_cumulative_delta_by_roi.png",
                                   excluded_subjects=excluded_present)

    # --- Loss table (unfiltered, on purpose) ---
    loss_table = build_loss_table(df_raw)
    print_loss_table(loss_table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate every fold even if cached results "
                             "already exist under results/loso_19_fold/analysis/cache/.")
    args = parser.parse_args()
    main(force=args.force_recompute)