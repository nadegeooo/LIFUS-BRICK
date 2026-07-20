"""
================================================================================
LOSO fold-specific ΔC (post - pre) by patient  —  reuses compare_pre_post.py
================================================================================

For each completed LOSO fold in results/loso_19_fold/fold_{subject}/, loads
that fold's best_model_cls.pt (the model trained with `subject` held out
entirely) and evaluates it ONLY on `subject`'s own pre/post data -- every
value plotted here is a genuine held-out evaluation, never data the
checkpoint was trained on.

Produces TWO separate figures (not one combined one, since a single figure
with both targets is too crowded):
    loso_delta_by_patient_vim.png
    loso_delta_by_patient_zi.png

Each is a 24-panel grid (one panel per ROI), x-axis = patient ID, 
y-axis = decoder-projected ΔC = post - pre (raw, non-normalized), 
ONE bar per patient. VIM plot bars are one color, ZI plot bars are 
the other (blue/orange).

Patient ordering (x-axis) on each plot is grouped by crossover order: o
n the VIM plot, patients whose group_str == "VIM_first" (VIM was their 
first treatment) are placed on the left; patients whose group_str ==
"ZI_first" (VIM was their second treatment) are placed on the right, 
separated by a dotted vertical line. The ZI plot mirrors this:
ZI_first on the left, VIM_first (ZI second) on the right. Within each group,
patients are sorted by ID.

A companion loss figure per target (loso_losses_by_patient_{target}.png)
shows the held-out reconstruction and classification loss, pre vs. post, per
patient, same ordering/grouping.

DATA LOADING: patient data is loaded directly through training.dataset.BRICKDataset. 
BRICKDataset's __getitem__ returns x already z-scored per-ROI exactly as used in 
training, plus subject_id, target, condition_str, group_str, first_target, 
lifus_condition.

REMAINING ASSUMPTIONS ABOUT compare_pre_post.py's PUBLIC API (still
unconfirmed -- source file wasn't available to check directly):
    - cpp.load_model(path) -> BRICK model, loaded + eval mode
    - cpp.compute_K(model) -> (K, Lambda, W_bar_x)
    - cpp.compute_roi_projection_weights(W_bar_x) -> roi_weights
    - cpp.project_to_roi(diag_matrix, roi_weights) -> roi-projected values,
      shape (n_items, len(TARGET_ROIS))
    - cpp.TARGET_ROIS -> ordered list of the 24 ROI names
    - model(x, label, kl_g0_weight=..., kl_u_weight=..., apply_free_bits=...)
      returns a dict with "C" (full MxM matrix) and "losses" ->
      {"loss_recon": ..., "loss_cls": ...} as scalar tensors, mirroring
      train.py's run_epoch() usage

Usage:
    python analysis/loso_analysis.py
    python analysis/loso_analysis.py --force-recompute
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
from training.dataset import BRICKDataset
from training.train import DATA_DIR

TARGETS = ["vim", "zi"]
TARGET_FIRST_LABEL = {"vim": "VIM_first", "zi": "ZI_first"}
TARGET_COLOR = {"vim": "#4C72B0", "zi": "#DD8452"}   # blue / orange

LOSO_DIR  = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2"
OUT_DIR   = LOSO_DIR / "results" / "loso_19_fold_beta_0.2"
CACHE_DIR = OUT_DIR / "cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)


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
            "session":         "pre" if item["condition_str"] == "mpre" else "post",
            "group_str":       item["group_str"],
            "x":               item["x"],
            "lifus_condition": int(item["lifus_condition"]),
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
    """
    my_items = [it for it in all_items if it["subject_id"] == subject_id]
    if not my_items:
        print(f"  WARNING: no data found for subject_id={subject_id} -- check "
              f"the ID format matches BRICKDataset exactly (e.g. 'sub-fuspd13').")
        return []
 
    model = cpp.load_model(checkpoint_path)
    K, Lambda, W_bar_x = cpp.compute_K(model)
    roi_weights = cpp.compute_roi_projection_weights(W_bar_x)
 
    rows = []
    with torch.no_grad():
        for target in TARGETS:
            pre_item  = next((it for it in my_items
                               if it["target"] == target and it["session"] == "pre"), None)
            post_item = next((it for it in my_items
                               if it["target"] == target and it["session"] == "post"), None)
            if pre_item is None or post_item is None:
                continue  # this subject doesn't have this target
 
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
                roi_c[session] = cpp.project_to_roi(C_diag.unsqueeze(0), roi_weights)[0]
 
                losses[session] = {
                    "loss_recon": out["losses"]["loss_recon"].item(),
                    "loss_cls":   out["losses"]["loss_cls"].item(),
                }
 
            delta = roi_c["post"] - roi_c["pre"]
 
            for roi_idx, roi_name in enumerate(cpp.TARGET_ROIS):
                rows.append({
                    "subject_id":       subject_id,
                    "target":           target,
                    "group_str":        group_str,
                    "roi":              roi_name,
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
def load_loso_data(force: bool = False) -> pd.DataFrame:
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
 
    df = pd.concat(frames, ignore_index=True)
    out_csv = OUT_DIR / "loso_delta_summary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved aggregated summary: {out_csv}")
    return df
 
 
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
# PLOT — ΔC grid, one figure per target
# ================================================================================
def compute_shared_ylim(df: pd.DataFrame, pad_frac: float = 0.1) -> tuple:
    max_abs = df["delta"].abs().max()
    if not np.isfinite(max_abs) or max_abs == 0:
        return (-1.0, 1.0)
    padded = max_abs * (1 + pad_frac)
    return (-padded, padded)
 
 
def plot_delta_grid(df: pd.DataFrame, target: str, ylim: tuple):
    sub = df[df["target"] == target]
    if sub.empty:
        print(f"  No data for target={target}, skipping plot.")
        return
 
    rois = list(cpp.TARGET_ROIS)
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
    fig.suptitle(
        f"LOSO held-out \u0394C by patient, per ROI \u2014 {target.upper()} treatment\n"
        f"left of dotted line: {first_label} (n={len(first_group)})   |   "
        f"right: {other_label} (n={len(second_group)})",
        fontsize=12, y=1.05
    )
    fig.tight_layout()
 
    out_path = OUT_DIR / f"loso_delta_by_patient_{target}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
 
 
# ================================================================================
# LOSS TABLE — one row per patient (not split by target)
# ================================================================================
def build_loss_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    One row per patient, averaging recon/cls loss across every available
    (target, session) evaluation for that patient's fold checkpoint --
    VIM-pre, VIM-post, ZI-pre, ZI-post, wherever present. The checkpoint is
    per-patient (not per-target), so VIM- and ZI-session losses for the same
    patient are expected to be close; this collapses them into a single
    per-patient number rather than reporting near-duplicate rows per target.
    """
    loss_cols = ["loss_recon_pre", "loss_recon_post", "loss_cls_pre", "loss_cls_post"]
    per_target = df.drop_duplicates(subset=["subject_id", "target"])[
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
    return table
 
 
def print_loss_table(table: pd.DataFrame):
    print(f"\nHeld-out loss per patient (n={len(table)}, averaged across "
          f"VIM/ZI and pre/post):")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
 
    out_csv = OUT_DIR / "loso_loss_by_patient.csv"
    table.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")
 
 
# ================================================================================
# ENTRY POINT
# ================================================================================
def main(force: bool = False):
    df = load_loso_data(force=force)
    ylim = compute_shared_ylim(df)  # shared across BOTH target figures, for comparability
    for target in TARGETS:
        plot_delta_grid(df, target, ylim=ylim)
 
    loss_table = build_loss_table(df)
    print_loss_table(loss_table)
 
 
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-recompute", action="store_true",
                        help="Re-evaluate every fold even if cached results "
                             "already exist under results/loso_19_fold/analysis/cache/.")
    args = parser.parse_args()
    main(force=args.force_recompute)