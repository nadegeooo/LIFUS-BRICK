# analysis/analysis_helper_functions.py
"""
Functions used by more than one analysis script (compare_pre_post,
compare_batch_size, compare_seed_effects, loso_analyze).

Section 1: extraction -- model loading, data verification, K, C.
Section 2: statistical tests and projections that operate on extracted K/C.

Functions used by only one script stay local to that script.
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

from models.brick import BRICK
from models.koopman_utils import compute_lambda
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS


# ================================================================================
# 1. EXTRACTION (model loading, data verification, K, C)
# ================================================================================

def load_model(checkpoint_path: Path) -> BRICK:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BRICK(use_control=ckpt["use_control"], use_ic=ckpt["use_ic"],
                  h=ckpt["h"], m=ckpt["m"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


def znorm(x: torch.Tensor) -> torch.Tensor:
    """Per-ROI z-score over time."""
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)


def verify_roi_consistency(subjects):
    """
    Confirms every subject's ROI order matches TARGET_ROIS exactly. If it
    doesn't, row i means a different region for some subject, and every
    spatial readout computed downstream is invalid for that subject.
    """
    target = list(TARGET_ROIS)
    for s in subjects:
        if list(s["roi_names"]) != target:
            raise ValueError(
                f"ROI order mismatch for {s['subject_id']}/{s['target']}: "
                f"subject roi_names != TARGET_ROIS. Spatial labels would be wrong."
            )
    print(f"ROI order verified consistent across {len(subjects)} subject-sessions.")


def compute_K(model: BRICK):
    """Reconstructs the shared Koopman operator K, its spectrum, and the mode readout."""
    with torch.no_grad():
        Lambda  = compute_lambda(model.nu_log, model.theta_log)   # (M,) complex
        P_inv   = model.P_inv                                      # (M, M) complex
        P       = torch.linalg.inv(P_inv)
        K       = P @ torch.diag(Lambda) @ P_inv                   # (M, M) complex
        W_bar_x = model.W_bar_x                                    # (N_ROIS, M) complex
    return K.detach().cpu().numpy(), Lambda.detach().cpu().numpy(), W_bar_x.detach().cpu().numpy()


def _normalize_order(group_str: str) -> str:
    """
    Maps the 'group' field (written by combine_and_align_rois.py from
    patient_treatment_mapping.xlsx) to 'vim_first' / 'zi_first'.

    Exact match against the two confirmed values. Raises on anything else
    rather than guessing.
    """
    known = {"VIM_first": "vim_first", "ZI_first": "zi_first"}
    if group_str not in known:
        raise ValueError(
            f"Unrecognized group value {group_str!r}. Expected one of "
            f"{list(known)}. Check patient_treatment_mapping.xlsx and "
            f"update _normalize_order if new values were added."
        )
    return known[group_str]


def extract_all_C(model, subjects=None) -> dict:
    """
    One forward pass per subject-session:

        {
          subject_id: {
            "order": "vim_first" | "zi_first",
            "vim":   {"pre": (M,) ndarray, "post": (M,) ndarray},
            "zi":    {"pre": (M,) ndarray, "post": (M,) ndarray},
          },
          ...
        }

    diag(C) values are raw g-space coordinates (real part). C is confirmed
    exactly diagonal and real, so this representation is lossless -- the
    full (N,N) matrix can be reconstructed via torch.diag(...) wherever
    needed (e.g. brain-space projection) without a second forward pass.
    Not ROI-labeled -- see project_to_roi for that.

    subjects: pass an already-loaded list from load_all() to skip a second
    disk read when the caller also needs it for verify_roi_consistency or
    similar. Loads internally if not provided.
    """
    if subjects is None:
        subjects = load_all()

    out = {}
    seen_order = {}

    with torch.no_grad():
        for s in subjects:
            sid = s["subject_id"]
            tgt = s["target"]

            x_pre = znorm(torch.tensor(s["mpre"], dtype=torch.float32))
            x_post = znorm(torch.tensor(s["mpost"], dtype=torch.float32))
            pre = np.real(model(x_pre)["C"].diag().cpu().numpy())
            post = np.real(model(x_post)["C"].diag().cpu().numpy())

            out.setdefault(sid, {})[tgt] = {"pre": pre, "post": post}

            order = _normalize_order(s["group"])
            if sid in seen_order and seen_order[sid] != order:
                raise ValueError(
                    f"Subject {sid} has conflicting group labels across "
                    f"sessions ({seen_order[sid]} vs {order}). Check "
                    f"patient_treatment_mapping.xlsx."
                )
            seen_order[sid] = order

    for sid, order in seen_order.items():
        out[sid]["order"] = order

    incomplete = [
        sid for sid, d in out.items()
        if not {"vim", "zi", "order"} <= d.keys()
    ]
    if incomplete:
        print(f"Incomplete session data for subjects: {incomplete} "
              f"(missing vim, zi, or order) -- callers should handle these "
              f"being absent from per-target comparisons.")

    return out


def reconstruct_C_dict(C_all: dict, target: str) -> dict:
    """
    Rebuild the {(subject_id, target): {"pre": (M,M), "post": (M,M)}} shape
    that check_per_subject_consistency_brain_space expects, from the
    diagonal vectors stored in extract_all_C's output. Lossless since C is
    confirmed diagonal.
    """
    return {
        (sid, target): {
            "pre":  torch.diag(torch.tensor(d[target]["pre"])),
            "post": torch.diag(torch.tensor(d[target]["post"])),
        }
        for sid, d in C_all.items()
        if target in d
    }


# ================================================================================
# 2. STATISTICAL TESTS / PROJECTIONS (operate on extracted K/C)
# ================================================================================

def compute_roi_projection_weights(W_bar_x: np.ndarray) -> np.ndarray:
    """
    Real, non-negative (N_ROIS, M) weights derived from the decoder, used to
    project a g-space diagonal vector (e.g. diag(C)) onto ROIs:

        weight[i, m] = |W_bar_x[i, m]|^2 / sum_m' |W_bar_x[i, m']|^2

    Row-normalized weighted average (not sum), so the projected quantity
    stays in the same units as the input vector.
    """
    power = np.abs(W_bar_x) ** 2
    row_sum = power.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    return power / row_sum


def project_to_roi(diag_arr: np.ndarray, roi_weights: np.ndarray) -> np.ndarray:
    """diag_arr: (n_items, M) -> (n_items, N_ROIS) via roi_weights."""
    return diag_arr @ roi_weights.T


def paired_tests_per_coordinate(pre, post, alpha=0.05) -> pd.DataFrame:
    """
    Per g-space coordinate: paired t-test pre vs post across subjects, plus
    Wilcoxon signed-rank as a non-parametric companion. FDR-corrected across
    coordinates. No roi_name column -- raw coordinates carry no ROI meaning.
    """
    N, Mc = pre.shape
    diff = post - pre
    t = np.zeros(Mc); p = np.ones(Mc); w = np.ones(Mc)
    for m in range(Mc):
        if np.allclose(diff[:, m], 0.0):
            t[m], p[m] = 0.0, 1.0
        else:
            t[m], p[m] = stats.ttest_rel(post[:, m], pre[:, m])
        try:
            w[m] = stats.wilcoxon(post[:, m], pre[:, m]).pvalue
        except ValueError:
            w[m] = 1.0

    _, p_fdr, _, _ = multipletests(p, method="fdr_bh")

    return pd.DataFrame({
        "coord_index": np.arange(Mc),
        "mean_pre":    pre.mean(0),
        "mean_post":   post.mean(0),
        "delta":       diff.mean(0),
        "t_statistic": t,
        "p_value":     p,
        "p_value_fdr": p_fdr,
        "wilcoxon_p":  w,
        "significant": p_fdr < alpha,
    })


def paired_tests_per_roi(pre_roi, post_roi, roi_names, alpha=0.05) -> pd.DataFrame:
    """
    Per ROI, on decoder-projected values. Same paired t-test + Wilcoxon +
    FDR structure as the coordinate-level test, but with a valid ROI label.
    """
    N, n_rois = pre_roi.shape
    diff = post_roi - pre_roi
    t = np.zeros(n_rois); p = np.ones(n_rois); w = np.ones(n_rois)
    for i in range(n_rois):
        if np.allclose(diff[:, i], 0.0):
            t[i], p[i] = 0.0, 1.0
        else:
            t[i], p[i] = stats.ttest_rel(post_roi[:, i], pre_roi[:, i])
        try:
            w[i] = stats.wilcoxon(post_roi[:, i], pre_roi[:, i]).pvalue
        except ValueError:
            w[i] = 1.0
    _, p_fdr, _, _ = multipletests(p, method="fdr_bh")

    return pd.DataFrame({
        "roi_index":   np.arange(n_rois),
        "roi_name":    list(roi_names),
        "mean_pre":    pre_roi.mean(0),
        "mean_post":   post_roi.mean(0),
        "delta":       diff.mean(0),
        "t_statistic": t,
        "p_value":     p,
        "p_value_fdr": p_fdr,
        "wilcoxon_p":  w,
        "significant": p_fdr < alpha,
    })


def norm_omnibus(pre, post):
    """Paired t-test on ||C||_F. Since C is diagonal, ||C||_F = sqrt(sum diag^2)."""
    pre_norm  = np.sqrt((pre ** 2).sum(1))
    post_norm = np.sqrt((post ** 2).sum(1))
    t, p = stats.ttest_rel(post_norm, pre_norm)
    try:
        w = stats.wilcoxon(post_norm, pre_norm).pvalue
    except ValueError:
        w = 1.0
    return float(t), float(p), float(w)


def check_per_subject_consistency_brain_space(
    C_dict: dict,
    model:  BRICK,
    target: str = "zi",
) -> pd.DataFrame:
    """
    For each ROI, computes the sign of delta-C projected to brain space per
    subject and reports how many subjects agree on direction.

        delta_C_brain = W_real @ (C_post - C_pre) @ W_real.T   (N_ROIS, N_ROIS)

    Takes the diagonal: change in each ROI's self-control.

    C_dict: {(subject_id, target): {"pre": (M,M) tensor, "post": (M,M) tensor}}
    """
    keys = sorted([k for k in C_dict.keys() if k[1] == target])
    n_subjects = len(keys)

    if n_subjects == 0:
        raise ValueError(f"No subjects found for target={target}")

    print(f"  Checking brain-space consistency across {n_subjects} subjects for target={target}")

    with torch.no_grad():
        W_real = model.W_bar_x.real.cpu().numpy()

    delta_brain_diag = []
    for k in keys:
        C_pre  = C_dict[k]["pre"].numpy()
        C_post = C_dict[k]["post"].numpy()
        delta_C = C_post - C_pre
        delta_C_brain = W_real @ delta_C @ W_real.T
        delta_brain_diag.append(np.diag(delta_C_brain))

    delta_brain_diag = np.stack(delta_brain_diag)

    n_positive  = (delta_brain_diag > 0).sum(axis=0)
    n_negative  = (delta_brain_diag < 0).sum(axis=0)
    n_consistent = np.maximum(n_positive, n_negative)
    consistent_direction = np.where(n_positive >= n_negative, "increase", "decrease")
    consistency_fraction = n_consistent / n_subjects
    mean_delta = delta_brain_diag.mean(axis=0)

    df = pd.DataFrame({
        "roi":                   TARGET_ROIS,
        "n_positive":            n_positive,
        "n_negative":            n_negative,
        "n_consistent":          n_consistent,
        "consistent_direction":  consistent_direction,
        "consistency_fraction":  consistency_fraction,
        "mean_delta":            mean_delta,
    })

    return df.sort_values("n_consistent", ascending=False).reset_index(drop=True)


def report_consistency_brain_space(consistency_df: pd.DataFrame, n_subjects: int, threshold: int = 15):
    """Prints a brain-space consistency report."""
    print(f"\n{'='*60}")
    print(f"Brain-space \u0394C consistency (threshold: {threshold}/{n_subjects})")
    print(f"{'='*60}")

    consistent = consistency_df[consistency_df["n_consistent"] >= threshold]

    if len(consistent) == 0:
        print(f"No ROIs show consistent direction in {threshold}+ subjects.")
    else:
        print(f"ROIs with consistent direction in {threshold}+ subjects: {len(consistent)}")
        for _, row in consistent.iterrows():
            print(
                f"  {row['roi']:<35} "
                f"{int(row['n_consistent'])}/{n_subjects} subjects "
                f"{row['consistent_direction']} | "
                f"mean \u0394C = {row['mean_delta']:+.4f}"
            )

    print(f"\nAll ROIs ranked by consistency:")
    print(f"{'ROI':<35} {'n_agree':>8} {'direction':>12} {'mean_delta':>12}")
    print("-" * 70)
    for _, row in consistency_df.iterrows():
        print(
            f"{row['roi']:<35} "
            f"{int(row['n_consistent']):>5}/{n_subjects:<3} "
            f"{row['consistent_direction']:>12} "
            f"{row['mean_delta']:>+12.4f}"
        )