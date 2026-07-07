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

def plot_var_connections(
    var_results: list,
    roi_idx:     int,
    roi_name:    str,
    t_out:       np.ndarray,
    p_fdr_out:   np.ndarray,
    sig_out:     np.ndarray,
    t_in:        np.ndarray,
    p_fdr_in:    np.ndarray,
    sig_in:      np.ndarray,
    out_path:    Path,
):
    """
    Plot outgoing and incoming connection t-statistics for the sonicated ROI.
    Significant connections marked in red.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8))
    fig.suptitle(f"VAR Connection Changes for {roi_name} (pre vs post)", fontsize=13)

    x = np.arange(len(TARGET_ROIS))

    for ax, t_stats, sig, title in [
        (axes[0], t_out, sig_out, f"Outgoing from {roi_name} (drives others)"),
        (axes[1], t_in,  sig_in,  f"Incoming to {roi_name} (driven by others)"),
    ]:
        colors = ["red" if s else "steelblue" for s in sig]
        ax.bar(x, t_stats, color=colors)
        ax.axhline(y=0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(TARGET_ROIS, rotation=90, fontsize=7)
        ax.set_ylabel("t-statistic")
        ax.set_title(title)
        ax.grid(alpha=0.3, axis="y")

        # Add legend
        from matplotlib.patches import Patch
        ax.legend(handles=[
            Patch(color="red",      label="significant (FDR)"),
            Patch(color="steelblue", label="non-significant"),
        ])

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved VAR connections plot to {out_path}")


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
        t_out, p_fdr_out, sig_out,
        t_in,  p_fdr_in,  sig_in,
        FIGURES_DIR / f"var_connections_{roi_name}{suffix}.png",
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

    main(roi_name=args.roi, target_filter=args.target, alpha=args.alpha)