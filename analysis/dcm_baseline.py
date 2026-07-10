# analysis/dcm_baseline.py
"""
================================================================================
VAR Baseline — Effective Connectivity Analysis
================================================================================

Description:
    Fits a first-order Vector Autoregression (VAR(1)) model to pre- and
    post-sonication BOLD timeseries for each subject, then tests whether
    effective connectivity changes after sonication using paired t-tests
    with Benjamini-Hochberg FDR correction.

    VAR(1) model:
        x_{t+1} = A_eff @ x_t + noise

    A_eff[i,j] = directed influence of region j on region i (j drives i).
    Fit by OLS using numpy.linalg.lstsq.

    Outgoing from ROI k: column A_eff[:, k] — how strongly k drives others
    Incoming to ROI k:   row    A_eff[k, :]  — how strongly others drive k

Outputs:
    results/final_model/var_statistical_results.csv
    results/final_model/figures_final_model/var_connections.png
    results/final_model/figures_final_model/comparison_table.csv

Usage:
    python analysis/dcm_baseline.py --roi lh_vim
    python analysis/dcm_baseline.py --roi lh_zi
    python analysis/dcm_baseline.py --roi lh_vim --target vim
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.power import TTestPower

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS

RESULTS_DIR = ROOT_DIR / "results" / "dcm_analysis"
BRICK_RESULTS_DIR = ROOT_DIR / "results" / "final_model"
FIGURES_DIR = ROOT_DIR / "results" / "dcm_analysis" / "figures_dcm"


# ================================================================================
# 1. FIT VAR
# ================================================================================

def fit_var(timeseries: np.ndarray) -> np.ndarray:
    """
    Fit a VAR(1) model to a BOLD timeseries using OLS.

        x_{t+1} = A_eff @ x_t + noise

    Solved via: A_eff = argmin ||X_future - X_past @ A_eff.T||^2

    Args:
        timeseries (np.ndarray): shape (T, N) — timepoints x ROIs

    Returns:
        A_eff (np.ndarray): shape (N, N) — effective connectivity matrix
                            A_eff[i,j] = influence of j on i
    """

    X_past   = timeseries[:-1, :]   # (T-1, N) — predictors
    X_future = timeseries[1:,  :]   # (T-1, N) — targets

    # Solve X_future = X_past @ A_eff.T via OLS
    # lstsq solves X_future = X_past @ B, so B = A_eff.T
    B, _, _, _ = np.linalg.lstsq(X_past, X_future, rcond=None)
    A_eff = B.T                      # (N, N)

    return A_eff


# ================================================================================
# 2. PREDICTION MSE
# ================================================================================

def predict_mse(A_eff: np.ndarray, timeseries: np.ndarray) -> float:
    """
    Compute one-step prediction MSE for a given A_eff matrix.

    Args:
        A_eff      (np.ndarray): shape (N, N)
        timeseries (np.ndarray): shape (T, N)

    Returns:
        float: mean squared prediction error
    """
    X_past   = timeseries[:-1, :]   # (T-1, N)
    X_future = timeseries[1:,  :]   # (T-1, N)
    X_pred   = X_past @ A_eff.T     # (T-1, N)
    return float(np.mean((X_future - X_pred) ** 2))


# ================================================================================
# 3. CONNECTION EXTRACTION
# ================================================================================

def get_outgoing(A_eff: np.ndarray, roi_idx: int) -> np.ndarray:
    """
    Get outgoing connections from a ROI — how strongly it drives others.
    = column of A_eff for that ROI (roi_idx is j in A_eff[i,j]).

    Args:
        A_eff   (np.ndarray): shape (N, N)
        roi_idx (int):        index of the ROI of interest

    Returns:
        np.ndarray: shape (N,)
    """
    return A_eff[:, roi_idx]


def get_incoming(A_eff: np.ndarray, roi_idx: int) -> np.ndarray:
    """
    Get incoming connections to a ROI — how strongly others drive it.
    = row of A_eff for that ROI (roi_idx is i in A_eff[i,j]).

    Args:
        A_eff   (np.ndarray): shape (N, N)
        roi_idx (int):        index of the ROI of interest

    Returns:
        np.ndarray: shape (N,)
    """
    return A_eff[roi_idx, :]


# ================================================================================
# 4. EXTRACT VAR FOR ALL SUBJECTS
# ================================================================================

def extract_all_var(target_filter: str = None) -> list:
    """
    Fit VAR(1) on pre and post BOLD for all subjects.

    Args:
        target_filter: 'vim', 'zi', or None for all

    Returns:
        list of dicts with keys:
            subject_id, target, A_pre, A_post
    """
    subjects = load_all()
    results  = []

    for s in subjects:
        if target_filter and s["target"] != target_filter:
            continue

        # Z-score per ROI (same as training)
        mpre  = s["mpre"].astype(np.float64)
        mpost = s["mpost"].astype(np.float64)
        mpre  = (mpre  - mpre.mean(axis=0))  / (mpre.std(axis=0)  + 1e-8)
        mpost = (mpost - mpost.mean(axis=0)) / (mpost.std(axis=0) + 1e-8)

        A_pre  = fit_var(mpre)
        A_post = fit_var(mpost)

        results.append({
            "subject_id": s["subject_id"],
            "target":     s["target"],
            "A_pre":      A_pre,
            "A_post":     A_post,
        })
        print(f"  Fitted VAR for ({s['subject_id']}, {s['target']})")

    print(f"Fitted VAR for {len(results)} subject-target pairs")
    return results


# ================================================================================
# 5. STATISTICAL ANALYSIS
# ================================================================================

def run_var_ttest(
    results:    list,
    roi_idx:    int,
    connection: str = "outgoing",
    alpha:      float = 0.05,
) -> tuple:
    """
    Paired t-test on VAR connections pre vs post for a given ROI.

    Args:
        results:    list from extract_all_var
        roi_idx:    index of the ROI of interest
        connection: 'outgoing' or 'incoming'
        alpha:      FDR threshold

    Returns:
        t_stats  (np.ndarray): shape (N_ROIS,)
        p_vals   (np.ndarray): shape (N_ROIS,)
        p_fdr    (np.ndarray): shape (N_ROIS,)
        significant (np.ndarray): shape (N_ROIS,) bool
    """
    get_conn = get_outgoing if connection == "outgoing" else get_incoming

    pre_conns  = np.stack([get_conn(r["A_pre"],  roi_idx) for r in results])
    post_conns = np.stack([get_conn(r["A_post"], roi_idx) for r in results])

    N = pre_conns.shape[1]
    t_stats = np.zeros(N)
    p_vals  = np.zeros(N)

    for j in range(N):
        t_stats[j], p_vals[j] = stats.ttest_rel(
            pre_conns[:, j], post_conns[:, j]
        )

    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    significant    = p_fdr < alpha

    return t_stats, p_vals, p_fdr, significant


# ================================================================================
# 6. COHEN'S D
# ================================================================================

def cohens_d(pre: np.ndarray, post: np.ndarray) -> np.ndarray:
    """
    Compute Cohen's d effect size for paired samples.

    d = mean(post - pre) / std(post - pre)

    Args:
        pre  (np.ndarray): shape (N_subjects, N_connections)
        post (np.ndarray): shape (N_subjects, N_connections)

    Returns:
        np.ndarray: shape (N_connections,)
    """
    diff = post - pre
    return diff.mean(axis=0) / (diff.std(axis=0) + 1e-8)


# ================================================================================
# 7. COMPARISON TABLE
# ================================================================================

def build_comparison_table(
    var_results: list,
    roi_idx:     int,
    target:      str,
    alpha:       float = 0.05,
) -> pd.DataFrame:
    """
    Build side-by-side BRICK vs VAR comparison table.
    Loads BRICK results from saved brick_summary_{target}.csv.
    """
    # Load BRICK summary
    brick_csv = BRICK_RESULTS_DIR / f"brick_summary_{target}.csv"
    
    brick_n_sig    = 0
    brick_median_d = 0.0
    if brick_csv.exists():
        brick_df       = pd.read_csv(brick_csv)
        brick_n_sig    = int(brick_df["n_sig_roi"].iloc[0])
        brick_median_d = float(brick_df["median_cohens_d"].iloc[0])
        print(f"Loaded BRICK summary from {brick_csv}")
    else:
        print(f"No BRICK summary found at {brick_csv} — run compare_pre_post.py first")

    # VAR stats
    t_out, p_out, p_fdr_out, sig_out = run_var_ttest(
        var_results, roi_idx, connection="outgoing", alpha=alpha
    )
    t_in, p_in, p_fdr_in, sig_in = run_var_ttest(
        var_results, roi_idx, connection="incoming", alpha=alpha
    )
    n_sig_var = int(sig_out.sum() + sig_in.sum())

    # Cohen's d for VAR
    pre_out  = np.stack([get_outgoing(r["A_pre"],  roi_idx) for r in var_results])
    post_out = np.stack([get_outgoing(r["A_post"], roi_idx) for r in var_results])
    d_out    = cohens_d(pre_out, post_out)

    pre_in   = np.stack([get_incoming(r["A_pre"],  roi_idx) for r in var_results])
    post_in  = np.stack([get_incoming(r["A_post"], roi_idx) for r in var_results])
    d_in     = cohens_d(pre_in, post_in)
    median_d_var = float(np.median(np.abs(np.concatenate([d_out, d_in]))))

    rows = [
        {
            "method":          "BRICK",
            "n_significant":   brick_n_sig,
            "median_cohens_d": brick_median_d,
            "interpretation":  "Decoder-projected ROI control gains pre vs post",
        },
        {
            "method":          "VAR",
            "n_significant":   n_sig_var,
            "median_cohens_d": median_d_var,
            "interpretation":  "Directed effective connectivity pre vs post",
        },
    ]

    return pd.DataFrame(rows)


# ================================================================================
# 8. PLOTTING
# ================================================================================

def per_connection_bh_thresholds(p_vals: np.ndarray, alpha: float, df: int):
    """
    Compute each connection's own BH-FDR threshold, in both p and t space.

    Each p-value's threshold depends on its RANK among all N tests:
        threshold_p(rank k) = (k / N) * alpha
    Converted to |t| via the inverse t-distribution (two-tailed) so it can
    be displayed alongside the observed t-statistic.

    Returns:
        thresh_p (np.ndarray): shape (N,) -- this connection's own p threshold
        thresh_t (np.ndarray): shape (N,) -- same threshold in |t| units
        ranks    (np.ndarray): shape (N,) -- 1-indexed rank (1 = smallest p)
    """
    N = len(p_vals)
    order = np.argsort(p_vals)          # indices sorted by p ascending
    ranks_by_sorted_pos = np.arange(1, N + 1)

    # Map rank back to original connection order
    ranks = np.empty(N, dtype=int)
    ranks[order] = ranks_by_sorted_pos

    thresh_p = (ranks / N) * alpha
    thresh_t = stats.t.ppf(1 - thresh_p / 2, df)

    return thresh_p, thresh_t, ranks


def plot_var_connections(
    var_results: list,
    roi_idx:     int,
    roi_name:    str,
    t_out:       np.ndarray,
    p_out:       np.ndarray,
    p_fdr_out:   np.ndarray,
    sig_out:     np.ndarray,
    t_in:        np.ndarray,
    p_in:        np.ndarray,
    p_fdr_in:    np.ndarray,
    sig_in:      np.ndarray,
    out_path:    Path,
    alpha:       float = 0.05,
):
    """
    Horizontal bar plot of mean paired difference (post - pre) in VAR
    connection strength for the sonicated ROI. Bars colored by FDR
    significance; each bar labeled with its observed t-statistic AND
    the |t| threshold that connection's own BH rank required to pass.
    """
    from matplotlib.patches import Patch

    out_path.parent.mkdir(parents=True, exist_ok=True)

    pre_out  = np.stack([get_outgoing(r["A_pre"],  roi_idx) for r in var_results])
    post_out = np.stack([get_outgoing(r["A_post"], roi_idx) for r in var_results])
    diff_out = (post_out - pre_out).mean(axis=0)

    pre_in   = np.stack([get_incoming(r["A_pre"],  roi_idx) for r in var_results])
    post_in  = np.stack([get_incoming(r["A_post"], roi_idx) for r in var_results])
    diff_in  = (post_in - pre_in).mean(axis=0)

    df = len(var_results) - 1

    fig, axes = plt.subplots(1, 2, figsize=(14, 10), sharey=True)
    fig.suptitle(f"VAR Connection Changes for {roi_name} (post \u2212 pre, mean paired diff)", fontsize=13)

    y = np.arange(len(TARGET_ROIS))

    for ax, diffs, t_stats, p_raw, sig, title in [
        (axes[0], diff_out, t_out, p_out, sig_out, f"Outgoing from {roi_name}"),
        (axes[1], diff_in,  t_in,  p_in,  sig_in,  f"Incoming to {roi_name}"),
    ]:
        colors = ["seagreen" if s else "indianred" for s in sig]

        ax.barh(y, diffs, color=colors)
        ax.axvline(x=0, color="black", linewidth=0.8)
        ax.set_yticks(y)
        ax.set_yticklabels(TARGET_ROIS, fontsize=7)
        ax.invert_yaxis()
        ax.set_xlabel("\u0394 connection strength (post \u2212 pre)")
        ax.set_title(title, fontsize=10)
        ax.grid(alpha=0.3, axis="x")

        xmin, xmax = ax.get_xlim()
        span = xmax - xmin
        offset = span * 0.02

        # Per-connection BH threshold, in both p and t space
        thresh_p, thresh_t, ranks = per_connection_bh_thresholds(p_raw, alpha, df)

        for yi, (d, t, tc) in enumerate(zip(diffs, t_stats, thresh_t)):
            label = f"t={t:.2f} (crit \u00b1{tc:.2f})"
            if d >= 0:
                ax.text(d + offset, yi, label, va="center", ha="left", fontsize=6)
            else:
                ax.text(d - offset, yi, label, va="center", ha="right", fontsize=6)

        ax.set_xlim(xmin - span * 0.12, xmax + span * 0.12)  # extra room for longer labels

    legend_handles = [
        Patch(color="seagreen",  label="significant (FDR < \u03b1)"),
        Patch(color="indianred", label="not significant"),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.03, 1, 1])
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved horizontal VAR connections plot to {out_path}")
    plt.close()


def plot_a_eff_heatmap(
    var_results: list,
    out_path:    Path,
    condition:   str = "pre",
):
    """
    Plot mean A_eff matrix across subjects as a heatmap.

    Args:
        var_results: list from extract_all_var
        out_path:    save path
        condition:   'pre' or 'post'
    """
    key = f"A_{condition}"
    mean_A = np.mean([r[key] for r in var_results], axis=0)

    fig, ax = plt.subplots(figsize=(12, 10))
    vmax = np.abs(mean_A).max()
    im = ax.imshow(mean_A, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks(range(len(TARGET_ROIS)))
    ax.set_yticks(range(len(TARGET_ROIS)))
    ax.set_xticklabels(TARGET_ROIS, rotation=90, fontsize=7)
    ax.set_yticklabels(TARGET_ROIS, fontsize=7)
    ax.set_title(f"Mean VAR A_eff — {condition} sonication", fontsize=12)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved A_eff heatmap to {out_path}")
    plt.close()


def plot_diff_heatmap_comparison(
    vim_results: list,
    zi_results:  list,
    out_path:    Path,
):
    """
    Raw mean(A_post - A_pre) heatmap, side-by-side for VIM and ZI.
    No per-edge standardization — this is the raw effect magnitude,
    shared color scale across both panels so relative magnitude is honest.
    """
    diff_vim = np.mean([r["A_post"] - r["A_pre"] for r in vim_results], axis=0)
    diff_zi  = np.mean([r["A_post"] - r["A_pre"] for r in zi_results],  axis=0)

    vmax = max(np.abs(diff_vim).max(), np.abs(diff_zi).max())

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))
    fig.suptitle("Raw mean \u0394A_eff (post \u2212 pre), shared color scale", fontsize=13)

    for ax, diff, title in [(axes[0], diff_vim, "VIM"), (axes[1], diff_zi, "ZI")]:
        im = ax.imshow(diff, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(TARGET_ROIS)))
        ax.set_yticks(range(len(TARGET_ROIS)))
        ax.set_xticklabels(TARGET_ROIS, rotation=90, fontsize=6)
        ax.set_yticklabels(TARGET_ROIS, fontsize=6)
        ax.set_title(f"{title} (n={len(vim_results) if title=='VIM' else len(zi_results)})", fontsize=11)

    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="\u0394 connection strength (raw)")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved raw diff heatmap comparison to {out_path}")
    plt.close()


from scipy.stats import binomtest

def sign_concordance_test(var_results: list, alpha: float = 0.05) -> dict:
    """
    Per-edge sign test: is the direction of (A_post - A_pre) consistent
    across subjects more often than chance (p=0.5), independent of magnitude?

    Returns dict with concordance (fraction agreeing with majority sign),
    raw p-values, FDR-corrected p-values, and significance mask — all (N,N).
    """
    diffs = np.stack([r["A_post"] - r["A_pre"] for r in var_results])  # (n_subj, N, N)
    n_subj = diffs.shape[0]
    N = diffs.shape[1]

    n_pos = (diffs > 0).sum(axis=0)
    n_neg = (diffs < 0).sum(axis=0)
    n_nonzero = n_pos + n_neg

    concordance = np.zeros((N, N))
    p_raw = np.ones((N, N))

    for i in range(N):
        for j in range(N):
            n = n_nonzero[i, j]
            if n == 0:
                continue
            k = max(n_pos[i, j], n_neg[i, j])
            concordance[i, j] = k / n
            p_raw[i, j] = binomtest(int(k), int(n), p=0.5, alternative="greater").pvalue

    # exclude diagonal from multiple-comparison correction
    mask = ~np.eye(N, dtype=bool)
    p_fdr = np.ones((N, N))
    _, p_fdr_flat, _, _ = multipletests(p_raw[mask], method="fdr_bh")
    p_fdr[mask] = p_fdr_flat

    significant = p_fdr < alpha

    return {
        "concordance": concordance,
        "p_raw": p_raw,
        "p_fdr": p_fdr,
        "significant": significant,
        "n_subjects": n_subj,
    }


def print_sign_concordance_summary(conc_result: dict, target_name: str, top_k: int = 10):
    """
    Print a summary of sign concordance results: overall stats plus the
    top-k most concordant edges (by FDR p-value) with ROI names.
    """
    concordance = conc_result["concordance"]
    p_fdr       = conc_result["p_fdr"]
    sig         = conc_result["significant"]
    n_subj      = conc_result["n_subjects"]

    N = concordance.shape[0]
    mask = ~np.eye(N, dtype=bool)

    print(f"\n{'='*60}")
    print(f"Sign Concordance — {target_name} (n={n_subj} subjects)")
    print(f"{'='*60}")
    print(f"  Edges tested (off-diagonal): {mask.sum()}")
    print(f"  Mean concordance:            {concordance[mask].mean():.3f}")
    print(f"  Edges FDR-significant:       {sig[mask].sum()} / {mask.sum()}")

    # top-k edges by lowest FDR p-value
    idx = np.dstack(np.unravel_index(np.argsort(np.where(mask, p_fdr, np.inf), axis=None), p_fdr.shape))[0]
    print(f"\n  Top {top_k} edges by concordance:")
    shown = 0
    for i, j in idx:
        if i == j:
            continue
        print(f"    {TARGET_ROIS[j]:>18s} -> {TARGET_ROIS[i]:<18s}  "
              f"concordance={concordance[i,j]:.2f}  p_fdr={p_fdr[i,j]:.4f}"
              f"{'  *' if sig[i,j] else ''}")
        shown += 1
        if shown >= top_k:
            break


def matrix_paired_ttest(var_results: list, alpha: float = 0.05):
    """
    Paired t-test on every off-diagonal edge of A_eff (post vs pre) across
    the full N x N connectivity matrix, with BH-FDR correction.
    """
    A_pre  = np.stack([r["A_pre"]  for r in var_results])  # (n_subj, N, N)
    A_post = np.stack([r["A_post"] for r in var_results])
    N = A_pre.shape[1]
    mask = ~np.eye(N, dtype=bool)

    t_stats = np.zeros((N, N))
    p_vals  = np.ones((N, N))
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            t_stats[i, j], p_vals[i, j] = stats.ttest_rel(A_pre[:, i, j], A_post[:, i, j])

    p_fdr = np.ones((N, N))
    _, p_fdr_flat, _, _ = multipletests(p_vals[mask], method="fdr_bh", alpha=alpha)
    p_fdr[mask] = p_fdr_flat

    return t_stats, p_vals, p_fdr, mask


def bh_critical_alpha(p_vals: np.ndarray, alpha: float) -> float:
    """
    The actual per-comparison alpha implied by BH-FDR: the largest
    threshold p(k) = (k/m)*alpha for which the k-th smallest p-value
    still clears it (the step-up critical value). If nothing clears
    (as when zero edges are significant), returns the strictest
    threshold the procedure would have allowed (rank 1, i.e. alpha/m) —
    this is derived from your real p-values via the BH rule itself,
    not assumed a priori like Bonferroni.
    """
    m = len(p_vals)
    sorted_p = np.sort(p_vals)
    ranks = np.arange(1, m + 1)
    thresholds = (ranks / m) * alpha
    passed = sorted_p <= thresholds
    if passed.any():
        k_max = np.max(np.where(passed)[0]) + 1
        return thresholds[k_max - 1]
    return thresholds[0]


def power_sensitivity_report(
    var_results:  list,
    target_name:  str,
    alpha:        float = 0.05,
    target_power: float = 0.8,
):
    """
    Runs the actual paired t-test + BH-FDR correction across all
    off-diagonal edges, then reports the minimum detectable Cohen's d
    at target_power using the BH-derived critical alpha (not a
    Bonferroni approximation), compared against the observed median |d|.
    """
    n_subj = len(var_results)
    N = len(TARGET_ROIS)
    mask = ~np.eye(N, dtype=bool)

    t_stats, p_vals, p_fdr, _ = matrix_paired_ttest(var_results, alpha=alpha)
    n_tests = int(mask.sum())
    n_sig   = int((p_fdr[mask] < alpha).sum())

    alpha_effective = bh_critical_alpha(p_vals[mask], alpha=alpha)

    power_analysis = TTestPower()
    required_d = power_analysis.solve_power(
        effect_size=None, nobs=n_subj, alpha=alpha_effective, power=target_power
    )

    A_pre  = np.stack([r["A_pre"]  for r in var_results])
    A_post = np.stack([r["A_post"] for r in var_results])
    diff = A_post - A_pre
    d_map = diff.mean(axis=0) / (diff.std(axis=0) + 1e-8)
    observed_median_d = float(np.median(np.abs(d_map[mask])))

    print(f"\n{'='*60}")
    print(f"Power / Sensitivity — {target_name} (n={n_subj} subjects)")
    print(f"{'='*60}")
    print(f"  Edges tested:              {n_tests}")
    print(f"  Edges FDR-significant:     {n_sig} / {n_tests}")
    print(f"  BH-derived critical alpha: {alpha_effective:.2e}")
    print(f"  Min detectable |d| at {int(target_power*100)}% power: {required_d:.3f}")
    print(f"  Observed median |d|:       {observed_median_d:.3f}")
    print(f"  --> {'UNDERPOWERED' if observed_median_d < required_d else 'adequately powered'} "
          f"for the observed effect size at this sample size / correction level")

    return {
        "required_d": required_d,
        "observed_median_d": observed_median_d,
        "n_tests": n_tests,
        "n_sig": n_sig,
        "alpha_effective": alpha_effective,
    }


# ================================================================================
# 9. MAIN
# ================================================================================

def main(roi_name: str = "lh_vim", target_filter: str = None, alpha: float = 0.05):
    suffix = f"_{target_filter}" if target_filter else ""

    print("=" * 60)
    print("VAR Baseline Analysis")
    print(f"ROI: {roi_name} | Target filter: {target_filter or 'all'}")
    print("=" * 60)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if roi_name not in TARGET_ROIS:
        raise ValueError(f"ROI '{roi_name}' not found. Available: {TARGET_ROIS}")
    roi_idx = TARGET_ROIS.index(roi_name)

    # Fit VAR for all subjects
    print("\nFitting VAR models...")
    var_results = extract_all_var(target_filter=target_filter)

    # Statistical analysis
    print("\nRunning statistical analysis...")
    t_out, p_out, p_fdr_out, sig_out = run_var_ttest(
        var_results, roi_idx, connection="outgoing", alpha=alpha
    )
    t_in, p_in, p_fdr_in, sig_in = run_var_ttest(
        var_results, roi_idx, connection="incoming", alpha=alpha
    )

    print(f"  Outgoing significant (FDR): {sig_out.sum()} / {len(sig_out)}")
    print(f"  Incoming significant (FDR): {sig_in.sum()} / {len(sig_in)}")

    # Save VAR statistical results
    df_out = pd.DataFrame({
        "roi":         TARGET_ROIS,
        "direction":   "outgoing",
        "t_statistic": t_out,
        "p_value":     p_out,
        "p_value_fdr": p_fdr_out,
        "significant": sig_out,
    })
    df_in = pd.DataFrame({
        "roi":         TARGET_ROIS,
        "direction":   "incoming",
        "t_statistic": t_in,
        "p_value":     p_in,
        "p_value_fdr": p_fdr_in,
        "significant": sig_in,
    })
    df_var = pd.concat([df_out, df_in], ignore_index=True)
    var_csv = RESULTS_DIR / f"var_statistical_results_{roi_name}{suffix}.csv"
    df_var.to_csv(var_csv, index=False)
    print(f"\nVAR results saved to {var_csv}")

    table = build_comparison_table(
        var_results, roi_idx,
        target=args.roi.replace("lh_", ""),   # e.g. "lh_vim" -> "vim"
        alpha=alpha,
    )
    print("\nComparison Table:")
    print(table.to_string(index=False))

    table_path = RESULTS_DIR / f"comparison_table_{roi_name}{suffix}.csv"
    table.to_csv(table_path, index=False)
    print(f"\nComparison table saved to {table_path}")

    # Plots
    plot_var_connections(
        var_results, roi_idx, roi_name,
        t_out, p_out, p_fdr_out, sig_out,
        t_in,  p_in,  p_fdr_in,  sig_in,
        FIGURES_DIR / f"var_connections_{roi_name}{suffix}.png",
        alpha=alpha,
    )
    plot_a_eff_heatmap(
        var_results,
        FIGURES_DIR / f"var_A_eff_pre_{roi_name}{suffix}.png",
        condition="pre",
    )
    plot_a_eff_heatmap(
        var_results,
        FIGURES_DIR / f"var_A_eff_post_{roi_name}{suffix}.png",
        condition="post",
    )

    if target_filter in ("vim", "zi"):
        conc = sign_concordance_test(var_results, alpha=alpha)
        print_sign_concordance_summary(conc, target_name=target_filter.upper())
        power_sensitivity_report(var_results, target_name=target_filter.upper(), alpha=alpha)

    return var_results, df_var, table


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--roi",    type=str, default="lh_vim",
                        help="ROI name to analyze. Default: lh_vim")
    parser.add_argument("--target", type=str, default=None,
                        choices=["vim", "zi"],
                        help="Filter by target. Default: all.")
    parser.add_argument("--alpha",  type=float, default=0.05,
                        help="FDR significance threshold. Default: 0.05.")
    args = parser.parse_args()

    var_results, df_var, table = main(roi_name=args.roi, target_filter=args.target, alpha=args.alpha)

    if args.target is None:
        print("\nGenerating combined VIM/ZI raw diff heatmap...")
        vim_results = [r for r in var_results if r["target"] == "vim"]
        zi_results  = [r for r in var_results if r["target"] == "zi"]
        plot_diff_heatmap_comparison(
            vim_results, zi_results,
            FIGURES_DIR / "var_diff_heatmap_vim_vs_zi.png",
        )