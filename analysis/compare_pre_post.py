# analysis/compare_pre_post.py
"""
================================================================================
Pre/Post Statistical Analysis
================================================================================

Description:
    Tests the primary scientific hypothesis: do latent dynamical modes change
    after sonication?

    For each subject and target (VIM/ZI):
        1. Load mpre and mpost from preprocessed .mat files
        2. Run through trained BRICK model to extract C_pre and C_post
        3. Test diagonal of C (control gains per latent dimension) pre vs post
           using paired t-tests across subjects
        4. Apply Benjamini-Hochberg FDR correction across all M modes
        5. Run paired t-test on Frobenius norm of C as an omnibus test

    Outputs:
        results/statistical_results.csv
        results/figures/koopman_spectrum.png
        results/figures/control_gains_pre_post.png

Usage:
    python analysis/compare_pre_post.py
    python analysis/compare_pre_post.py --target vim
    python analysis/compare_pre_post.py --target zi
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from models.brick import BRICK
from models.koopman_utils import compute_lambda
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS

FINAL_MODEL_PATH = ROOT_DIR / "results" / "final_model" / "best_model.pt"
RESULTS_DIR      = ROOT_DIR / "results" / "final_model"
FIGURES_DIR      = ROOT_DIR / "results" / "final_model" / "figures_final_model"


# ================================================================================
# 1. LOAD MODEL
# ================================================================================

def load_model(checkpoint_path: Path = FINAL_MODEL_PATH) -> BRICK:
    """Load BRICK model from checkpoint."""
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model = BRICK(
        use_control=checkpoint["use_control"],
        use_ic=checkpoint["use_ic"],
        h=checkpoint["h"],
        m=checkpoint["m"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    print(f"  use_control={checkpoint['use_control']}, use_ic={checkpoint['use_ic']}, "
          f"h={checkpoint['h']}, m={checkpoint['m']}")
    return model


# ================================================================================
# 2. EXTRACT C MATRICES
# ================================================================================

def extract_C_matrices(model: BRICK, target_filter: str = None) -> dict:
    """
    Run inference on all subjects and extract C_pre and C_post.

    Args:
        model:         Trained BRICK model in eval mode
        target_filter: If 'vim' or 'zi', only process that target.
                       If None, process all.

    Returns:
        dict keyed by (subject_id, target) -> {"pre": C_pre, "post": C_post}
        where C_pre and C_post are tensors of shape (M, M)
    """
    subjects = load_all()
    C_dict = {}

    with torch.no_grad():
        for s in subjects:
            if target_filter and s["target"] != target_filter:
                continue

            x_pre  = torch.tensor(s["mpre"],  dtype=torch.float32)
            x_post = torch.tensor(s["mpost"], dtype=torch.float32)

            # Z-score normalize to self mean and std (same as training)
            x_pre  = (x_pre  - x_pre.mean(dim=0))  / (x_pre.std(dim=0)  + 1e-8)
            x_post = (x_post - x_post.mean(dim=0)) / (x_post.std(dim=0) + 1e-8)

            C_pre  = model(x_pre)["C"]
            C_post = model(x_post)["C"]

            key = (s["subject_id"], s["target"])
            C_dict[key] = {"pre": C_pre, "post": C_post}
            print(f"  Extracted C for {key}")

    print(f"Extracted C matrices for {len(C_dict)} subject-target pairs")
    return C_dict


# ================================================================================
# 3. KOOPMAN MODES
# ================================================================================

def compute_koopman_modes(model: BRICK) -> dict:
    """
    Extract Koopman eigenvalues and compute brain space projections.

    The learned K = P @ diag(Lambda) @ P_inv.
    Each eigenvector (column of P) projected through W_bar_x gives
    a spatial loading over N_ROIS brain regions.

    Returns dict with:
        eigenvalues  (M,)     complex — Lambda
        magnitudes   (M,)     real    — |Lambda| (decay rates)
        phases       (M,)     real    — angle(Lambda) (oscillatory frequencies)
        brain_maps   (M, N_ROIS) real — spatial loadings per mode
        sort_order   (M,)     int     — indices sorted by magnitude descending
    """
    with torch.no_grad():
        Lambda = compute_lambda(model.nu_log, model.theta_log)  # (M,) complex
        magnitudes = Lambda.abs().cpu().numpy()                  # (M,) real
        phases     = Lambda.angle().cpu().numpy()                # (M,) real

        # Brain space projection: W_bar_x @ e_j where e_j is j-th standard basis
        # = j-th column of W_bar_x, real part
        W_bar_x   = model.W_bar_x.cpu()                         # (N_ROIS, M) complex
        brain_maps = W_bar_x.real.numpy()                        # (N_ROIS, M) real
        brain_maps = brain_maps.T                                # (M, N_ROIS)

    sort_order = np.argsort(magnitudes)[::-1]  # descending by magnitude

    return {
        "eigenvalues": Lambda.cpu().numpy(),
        "magnitudes":  magnitudes,
        "phases":      phases,
        "brain_maps":  brain_maps,
        "sort_order":  sort_order,
    }


# ================================================================================
# 4. BRAIN SPACE
# ================================================================================
def project_to_brain_space(model: BRICK) -> dict:
    """
    Project K and shared model components back to 24-ROI brain space.

    K_brain  = W_real @ K.real @ W_real.T   (24, 24)
    |K_brain| = W_real @ |K|   @ W_real.T   (24, 24)

    where W_real = W_bar_x.real (24, 96)

    Returns dict with:
        K_brain_real  (24, 24) — real part of K in brain space
        K_brain_mag   (24, 24) — magnitude of K in brain space
        W_real        (24, 96) — projection matrix
    """
    with torch.no_grad():
        Lambda = compute_lambda(model.nu_log, model.theta_log)  # (M,) complex
        P_inv  = model.P_inv.cpu()                               # (M, M) complex
        W_bar_x = model.W_bar_x.cpu()                           # (N, M) complex
        W_real  = W_bar_x.real                                   # (N, M) real

        # Reconstruct P = inv(P_inv)
        P = torch.linalg.inv(P_inv)                             # (M, M) complex

        # Reconstruct K = P @ diag(Lambda) @ P_inv
        K = P @ torch.diag(Lambda) @ P_inv                      # (M, M) complex

        # Project to brain space
        W_np      = W_real.numpy()                               # (24, 96)
        K_real_np = K.real.numpy()                               # (96, 96)
        K_mag_np  = K.abs().numpy()                              # (96, 96)

        K_brain_real = W_np @ K_real_np @ W_np.T                # (24, 24)
        K_brain_mag  = W_np @ K_mag_np  @ W_np.T                # (24, 24)

    return {
        "K_brain_real": K_brain_real,
        "K_brain_mag":  K_brain_mag,
        "W_real":       W_np,
    }


def project_C_to_brain_space(C_dict: dict, model: BRICK) -> dict:
    """
    Project per-subject C matrices to brain space and compute
    mean pre, mean post, and mean delta (post - pre).

    C_brain = W_real @ C @ W_real.T   (24, 24)

    Returns dict with:
        mean_pre   (24, 24)
        mean_post  (24, 24)
        mean_delta (24, 24)
    """
    with torch.no_grad():
        W_real = model.W_bar_x.real.cpu().numpy()   # (24, 96)

    pre_maps  = []
    post_maps = []

    for key, val in C_dict.items():
        C_pre  = val["pre"].numpy()                  # (96, 96)
        C_post = val["post"].numpy()                 # (96, 96)

        C_pre_brain  = W_real @ C_pre  @ W_real.T   # (24, 24)
        C_post_brain = W_real @ C_post @ W_real.T   # (24, 24)

        pre_maps.append(C_pre_brain)
        post_maps.append(C_post_brain)

    mean_pre   = np.mean(pre_maps,  axis=0)          # (24, 24)
    mean_post  = np.mean(post_maps, axis=0)          # (24, 24)
    mean_delta = mean_post - mean_pre                 # (24, 24)

    return {
        "mean_pre":   mean_pre,
        "mean_post":  mean_post,
        "mean_delta": mean_delta,
    }


def plot_brain_space_heatmaps(
    K_maps: dict,
    C_maps: dict,
    roi_names: list,
    out_dir: Path,
    suffix: str = "",
):
    """
    Plot K and C brain space heatmaps with ROI names on axes.

    Produces:
        K_brain_real.png  — real part of K
        K_brain_mag.png   — magnitude of K
        C_brain_pre.png   — mean C pre sonication
        C_brain_post.png  — mean C post sonication
        C_brain_delta.png — mean delta C (post - pre)
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    def save_heatmap(matrix, title, filename, cmap="RdBu_r", center_zero=True):
        fig, ax = plt.subplots(figsize=(12, 10))
        vmax = np.abs(matrix).max()
        vmin = -vmax if center_zero else matrix.min()

        im = ax.imshow(matrix, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        ax.set_xticks(range(len(roi_names)))
        ax.set_yticks(range(len(roi_names)))
        ax.set_xticklabels(roi_names, rotation=90, fontsize=7)
        ax.set_yticklabels(roi_names, fontsize=7)
        ax.set_title(title, fontsize=12)

        plt.tight_layout()
        path = out_dir / f"{filename}{suffix}.png"
        plt.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved {path}")
        plt.close()

    save_heatmap(
        K_maps["K_brain_real"],
        "Koopman Operator K — Real Part (Brain Space)",
        "K_brain_real",
        cmap="RdBu_r",
        center_zero=True,
    )
    save_heatmap(
        K_maps["K_brain_mag"],
        "Koopman Operator K — Magnitude (Brain Space)",
        "K_brain_mag",
        cmap="hot",
        center_zero=False,
    )
    save_heatmap(
        C_maps["mean_pre"],
        "Control Matrix C — Mean Pre Sonication (Brain Space)",
        "C_brain_pre",
        cmap="RdBu_r",
        center_zero=True,
    )
    save_heatmap(
        C_maps["mean_post"],
        "Control Matrix C — Mean Post Sonication (Brain Space)",
        "C_brain_post",
        cmap="RdBu_r",
        center_zero=True,
    )
    save_heatmap(
        C_maps["mean_delta"],
        "Control Matrix ΔC — Mean Post minus Pre (Brain Space)",
        "C_brain_delta",
        cmap="RdBu_r",
        center_zero=True,
    )


# ================================================================================
# 5. STATISTICAL ANALYSIS
# ================================================================================

def run_statistical_analysis(C_dict: dict, alpha: float = 0.05) -> pd.DataFrame:
    """
    For each latent dimension m, extract the diagonal control gain C[m,m]
    for pre and post across all subjects, then run a paired t-test.
    Apply Benjamini-Hochberg FDR correction across all M modes.

    Args:
        C_dict: dict from extract_C_matrices
        alpha:  significance threshold after FDR correction

    Returns:
        DataFrame with columns:
            mode_index, eigenvalue_mag, eigenvalue_phase,
            t_statistic, p_value, p_value_fdr, significant
    """
    keys  = sorted(C_dict.keys())
    M     = next(iter(C_dict.values()))["pre"].shape[0]

    # Stack diagonal control gains: shape (n_subjects, M)
    pre_gains  = np.stack([C_dict[k]["pre"].diag().numpy()  for k in keys])  # (N, M)
    post_gains = np.stack([C_dict[k]["post"].diag().numpy() for k in keys])  # (N, M)

    # Paired t-test per mode
    t_stats = np.zeros(M)
    p_vals  = np.zeros(M)
    for m in range(M):
        t_stats[m], p_vals[m] = stats.ttest_rel(pre_gains[:, m], post_gains[:, m])

    # FDR correction
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    significant    = p_fdr < alpha

    # Placeholder eigenvalue info (filled in after compute_koopman_modes)
    df = pd.DataFrame({
        "mode_index":       np.arange(M),
        "eigenvalue_mag":   np.zeros(M),   # filled below
        "eigenvalue_phase": np.zeros(M),   # filled below
        "t_statistic":      t_stats,
        "p_value":          p_vals,
        "p_value_fdr":      p_fdr,
        "significant":      significant,
    })

    return df


def add_eigenvalue_info(df: pd.DataFrame, modes: dict) -> pd.DataFrame:
    """Add eigenvalue magnitudes and phases to results dataframe."""
    df = df.copy()
    df["eigenvalue_mag"]   = modes["magnitudes"]
    df["eigenvalue_phase"] = modes["phases"]
    return df


def run_c_norm_test(C_dict: dict) -> tuple:
    """
    Omnibus test: paired t-test on Frobenius norm of C pre vs post.

    Returns:
        t_stat (float), p_value (float)
    """
    keys       = sorted(C_dict.keys())
    pre_norms  = np.array([C_dict[k]["pre"].norm().item()  for k in keys])
    post_norms = np.array([C_dict[k]["post"].norm().item() for k in keys])
    t, p       = stats.ttest_rel(pre_norms, post_norms)
    return float(t), float(p)


# ================================================================================
# 6. PLOTTING
# ================================================================================

def plot_koopman_spectrum(modes: dict, out_path: Path):
    """Plot eigenvalue magnitudes sorted by mode index."""
    sort_order = modes["sort_order"]
    magnitudes = modes["magnitudes"][sort_order]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle("Koopman Spectrum", fontsize=13)

    axes[0].bar(range(len(magnitudes)), magnitudes)
    axes[0].set_xlabel("Mode rank (sorted by magnitude)")
    axes[0].set_ylabel("|Lambda|")
    axes[0].set_title("Eigenvalue Magnitudes")
    axes[0].axhline(y=1.0, color='red', linestyle='--', label='unit circle')
    axes[0].legend()

    # Phase plot
    phases = modes["phases"][sort_order]
    axes[1].scatter(magnitudes * np.cos(phases),
                    magnitudes * np.sin(phases), alpha=0.6)
    axes[1].set_xlabel("Real part")
    axes[1].set_ylabel("Imaginary part")
    axes[1].set_title("Eigenvalues in Complex Plane")
    circle = plt.Circle((0, 0), 1, fill=False, color='red', linestyle='--')
    axes[1].add_patch(circle)
    axes[1].set_aspect('equal')

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved spectrum plot to {out_path}")
    plt.show()


def plot_control_gains(C_dict: dict, df: pd.DataFrame, out_path: Path):
    """
    Scatter plot of mean pre vs mean post diagonal control gains per mode.
    Points above diagonal = increased post-sonication.
    Points below diagonal = decreased post-sonication.
    Significant modes marked in red.
    """
    keys       = sorted(C_dict.keys())
    pre_gains  = np.stack([C_dict[k]["pre"].diag().numpy()  for k in keys])
    post_gains = np.stack([C_dict[k]["post"].diag().numpy() for k in keys])

    mean_pre  = pre_gains.mean(axis=0)   # (M,)
    mean_post = post_gains.mean(axis=0)  # (M,)
    sig_mask  = df["significant"].values

    fig, ax = plt.subplots(figsize=(7, 7))

    # Non-significant modes
    ax.scatter(mean_pre[~sig_mask], mean_post[~sig_mask],
               alpha=0.5, color="steelblue", label="non-significant", s=30)

    # Significant modes
    if sig_mask.any():
        ax.scatter(mean_pre[sig_mask], mean_post[sig_mask],
                   alpha=0.9, color="red", label="significant (FDR)", s=60, zorder=5)
        for i in np.where(sig_mask)[0]:
            ax.annotate(str(i), (mean_pre[i], mean_post[i]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")

    # Diagonal reference line (no change)
    lim = max(np.abs(mean_pre).max(), np.abs(mean_post).max()) * 1.1
    ax.plot([-lim, lim], [-lim, lim], 'k--', linewidth=1, label="no change")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    ax.set_xlabel("Mean diagonal C — pre sonication")
    ax.set_ylabel("Mean diagonal C — post sonication")
    ax.set_title("Control Gains Pre vs Post Sonication\n(each point = one Koopman mode)")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefms(out_path, dpi=150)
    print(f"Saved control gains plot to {out_path}")
    plt.show()


# ================================================================================
# 7. MAIN
# ================================================================================

def main(target_filter: str = None, alpha: float = 0.05):
    suffix = f"_{target_filter}" if target_filter else ""

    print("=" * 60)
    print("BRICK Pre/Post Statistical Analysis")
    print(f"Target filter: {target_filter or 'all'}")
    print("=" * 60)

    FINAL_MODEL_DIR = ROOT_DIR / "results" / "final_model"
    FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # Load model
    model = load_model()

    # Extract C matrices
    print("\nExtracting C matrices...")
    C_dict = extract_C_matrices(model, target_filter=target_filter)

    # Koopman modes
    print("\nComputing Koopman modes...")
    modes = compute_koopman_modes(model)
    print(f"  Eigenvalue magnitudes: min={modes['magnitudes'].min():.4f}, "
          f"max={modes['magnitudes'].max():.4f}, "
          f"mean={modes['magnitudes'].mean():.4f}")

    # Statistical analysis
    print("\nRunning statistical analysis...")
    df = run_statistical_analysis(C_dict, alpha=alpha)
    df = add_eigenvalue_info(df, modes)

    n_sig = df["significant"].sum()
    print(f"  Modes surviving FDR correction (alpha={alpha}): {n_sig} / {len(df)}")

    # C norm omnibus test
    t_norm, p_norm = run_c_norm_test(C_dict)
    print(f"\nC norm paired t-test: t={t_norm:.4f}, p={p_norm:.4f}")

    # Save results CSV
    csv_path = RESULTS_DIR / f"statistical_results{suffix}.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Save brain maps for all modes
    brain_maps_df = pd.DataFrame(
        modes["brain_maps"],
        columns=TARGET_ROIS,
    )
    brain_maps_df.insert(0, "mode_index", np.arange(len(modes["brain_maps"])))
    brain_maps_df.insert(1, "eigenvalue_mag", modes["magnitudes"])
    brain_maps_df.insert(2, "eigenvalue_phase", modes["phases"])
    brain_maps_path = RESULTS_DIR / f"brain_maps{suffix}.csv"
    brain_maps_df.to_csv(brain_maps_path, index=False)
    print(f"Brain maps saved to {brain_maps_path}")

    # Print summary table
    print("\nTop 10 modes by |t-statistic|:")
    top = df.reindex(df["t_statistic"].abs().sort_values(ascending=False).index).head(10)
    print(top[["mode_index", "eigenvalue_mag", "t_statistic",
               "p_value", "p_value_fdr", "significant"]].to_string(index=False))

    # Significant modes with top brain regions
    if n_sig > 0:
        print(f"\nSignificant modes and top brain regions:")
        sig_modes = df[df["significant"]].sort_values("p_value_fdr")
        for _, row in sig_modes.iterrows():
            m_idx = int(row["mode_index"])
            loadings = modes["brain_maps"][m_idx]
            top_rois = np.argsort(np.abs(loadings))[::-1][:5]
            roi_names = [TARGET_ROIS[i] for i in top_rois]
            print(f"  Mode {m_idx:3d}: t={row['t_statistic']:+.3f}, "
                  f"p_fdr={row['p_value_fdr']:.4f} | "
                  f"top ROIs: {roi_names}")
    else:
        print("\nNo modes survived FDR correction.")

    # Plots
    plot_koopman_spectrum(modes, FIGURES_DIR / f"koopman_spectrum{suffix}.png")
    plot_control_gains(C_dict, df, FIGURES_DIR / f"control_gains{suffix}.png")

    # Brain space projections
    print("\nProjecting to brain space...")
    K_maps = project_to_brain_space(model)
    C_maps = project_C_to_brain_space(C_dict, model)
    plot_brain_space_heatmaps(
        K_maps, C_maps, TARGET_ROIS,
        out_dir=FIGURES_DIR,
        suffix=suffix,
    )

    return df, modes, C_dict


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None,
                        choices=["vim", "zi"],
                        help="Filter by target (vim or zi). Default: all.")
    parser.add_argument("--alpha",  type=float, default=0.05,
                        help="FDR significance threshold. Default: 0.05.")
    args = parser.parse_args()

    main(target_filter=args.target, alpha=args.alpha)