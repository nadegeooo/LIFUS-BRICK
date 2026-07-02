"""
================================================================================
Pre/Post Analysis  —  K (descriptive) and C (inferential)
================================================================================

Design (settled, do not pool the two halves):

    K is a GLOBAL parameter. nu_log, theta_log, P_inv, W_bar_x are all shared
    across the cohort, so K = P @ diag(Lambda) @ P_inv is ONE operator with no
    pre and no post. Nothing on K is tested. The K outputs are descriptive only
    and describe the cohort's baseline shared dynamics.

    C is the ONLY thing that varies by session, so it carries the entire
    sonication effect. C is diagonal by construction, so the per-subject object
    is the length-M vector diag(C). The hypothesis test is a per-coordinate
    paired t-test (pre vs post), run SEPARATELY for VIM and ZI (pooling the two
    targets per subject is pseudoreplication), FDR-corrected across coordinates.

Two bases, related by P (do not cross them):
    - g-space   : where g_0, C, K live. Coordinate m = (ROI m//H, channel m%H),
                  ROI-major, confirmed from Encoder.encode_distribution.
    - eigenbasis: where the dynamics are diagonal. Lambda and the mode maps
                  (columns of W_bar_x) live here.
    => diag(C)[m] is a g-space coordinate. It is NOT paired with Lambda[m].
       The C results table carries ROI/channel labels, never eigenvalues.

Units (TR = 2 s):
    discrete-time eigenvalue Lambda per step.
    |Lambda|    = persistence per TR        (1 = neutral, <1 decays)
    arg(Lambda) = oscillation rad per TR
    freq_hz     = |arg| / (2*pi*TR)
    period_s    = 2*pi*TR / |arg|
    tau_s       = -TR / ln|Lambda|          (decay time constant)

Usage:
    python analysis/compare_pre_post.py                 # both targets
    python analysis/compare_pre_post.py --target vim
    python analysis/compare_pre_post.py --top-k 8
"""

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import torch
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from config import M, N_ROIS, H
from models.brick import BRICK
from models.koopman_utils import compute_lambda
from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS

FINAL_MODEL_PATH = ROOT_DIR / "results" / "final_model" / "best_model.pt"
RESULTS_DIR      = ROOT_DIR / "results" / "final_model"
FIGURES_DIR      = ROOT_DIR / "results" / "final_model" / "figures_final_model"

TR = 2.0                       # seconds per volume
RESTING_BAND = (0.01, 0.10)    # Hz, conventional resting-state BOLD band
NYQUIST_HZ = 1.0 / (2 * TR)    # 0.25 Hz


# ================================================================================
# UTILITIES
# ================================================================================
 
def znorm(x: torch.Tensor) -> torch.Tensor:
    """Per-ROI z-score over time. Matches the training-time normalization."""
    return (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)
 
 
def eig_units(Lambda: np.ndarray) -> dict:
    """Convert complex eigenvalues to interpretable physical quantities."""
    mag    = np.abs(Lambda)
    phase  = np.angle(Lambda)
    aphase = np.abs(phase)
    freq_hz  = aphase / (2 * np.pi * TR)
    period_s = np.where(aphase > 1e-12, 2 * np.pi * TR / aphase, np.inf)
    with np.errstate(divide="ignore"):
        tau_s = np.where(mag < 1.0, -TR / np.log(mag), np.inf)
    return {"mag": mag, "phase": phase, "freq_hz": freq_hz,
            "period_s": period_s, "tau_s": tau_s}
 
 
def verify_roi_consistency(subjects):
    """
    Every subject's ROI order must be byte-identical to TARGET_ROIS. If not,
    row i meant a different region for some subject during training and every
    spatial readout below is invalid. Fail loudly rather than silently mislabel.
    """
    target = list(TARGET_ROIS)
    for s in subjects:
        if list(s["roi_names"]) != target:
            raise ValueError(
                f"ROI order mismatch for {s['subject_id']}/{s['target']}: "
                f"subject roi_names != TARGET_ROIS. Spatial labels would be wrong."
            )
    print(f"ROI order verified consistent across {len(subjects)} subject-sessions.")
 
 
def compute_roi_projection_weights(W_bar_x: np.ndarray) -> np.ndarray:
    """
    Real, non-negative (N_ROIS, M) weights derived from the decoder, used to
    project any g-space diagonal vector (e.g. diag(C)) onto ROIs:
 
        weight[i, m] = |W_bar_x[i, m]|^2 / sum_m' |W_bar_x[i, m']|^2
 
    This is diag(W diag(v) W^H) expressed as a row-normalized weighted AVERAGE
    rather than a weighted sum, so the projected quantity stays in the same
    units as v (a control gain) instead of being scaled by each ROI's total
    spectral power in the readout. This is the only valid way to attach a
    g-space diagonal vector to ROIs when the vector's own coordinates were
    never architecturally anchored to ROIs (true for diag(C); see module
    docstring).
    """
    power = np.abs(W_bar_x) ** 2                    # (N_ROIS, M) real, >=0
    row_sum = power.sum(axis=1, keepdims=True)
    row_sum = np.where(row_sum > 1e-12, row_sum, 1.0)
    return power / row_sum
 
 
def project_to_roi(diag_arr: np.ndarray, roi_weights: np.ndarray) -> np.ndarray:
    """diag_arr: (n_subjects, M) -> (n_subjects, N_ROIS) via roi_weights."""
    return diag_arr @ roi_weights.T


# ================================================================================
# 1. LOAD MODEL
# ================================================================================

def load_model(checkpoint_path: Path = FINAL_MODEL_PATH) -> BRICK:
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    model = BRICK(use_control=ckpt["use_control"], use_ic=ckpt["use_ic"],
                  h=ckpt["h"], m=ckpt["m"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded model from {checkpoint_path}")
    return model


# ================================================================================
# 2. K (GLOBAL, DESCRIPTIVE — NO STATISTICS)
# ================================================================================

def compute_K(model: BRICK):
    """Reconstruct the shared operator K, its spectrum, and the mode readout."""
    with torch.no_grad():
        Lambda  = compute_lambda(model.nu_log, model.theta_log)   # (M,) complex
        P_inv   = model.P_inv                                      # (M, M) complex
        P       = torch.linalg.inv(P_inv)
        K       = P @ torch.diag(Lambda) @ P_inv                   # (M, M) complex
        W_bar_x = model.W_bar_x                                    # (N_ROIS, M) complex
    return K.detach().cpu().numpy(), Lambda.detach().cpu().numpy(), W_bar_x.detach().cpu().numpy()


def eigenvalue_table(Lambda: np.ndarray) -> pd.DataFrame:
    u = eig_units(Lambda)
    df = pd.DataFrame({
        "mode_index":     np.arange(len(Lambda)),
        "eigenvalue_mag": u["mag"],
        "phase_rad":      u["phase"],
        "freq_hz":        u["freq_hz"],
        "period_s":       u["period_s"],
        "tau_s":          u["tau_s"],
    })
    return df.sort_values("eigenvalue_mag", ascending=False).reset_index(drop=True)


def plot_spectrum(Lambda: np.ndarray, out_path: Path):
    u = eig_units(Lambda)
    order = np.argsort(u["mag"])[::-1]
    mag, phase, freq = u["mag"][order], u["phase"][order], u["freq_hz"][order]

    fig, axes = plt.subplots(1, 3, figsize=(17, 4.5))

    axes[0].bar(range(len(mag)), mag, color="steelblue")
    axes[0].axhline(1.0, color="red", ls="--", lw=1, label="unit circle (no decay)")
    axes[0].set_xlabel("Mode (ranked by persistence)")
    axes[0].set_ylabel("|\u039b|")
    axes[0].set_title("Eigenvalue magnitude |\u039b|: persistence per TR\n"
                      "(1.0 = no decay, lower = faster decay)", fontsize=10)
    axes[0].legend(fontsize=8)

    axes[1].scatter(mag * np.cos(phase), mag * np.sin(phase),
                    alpha=0.6, color="steelblue")
    axes[1].add_patch(plt.Circle((0, 0), 1, fill=False, color="red", ls="--"))
    axes[1].set_aspect("equal")
    axes[1].set_xlabel("Re(\u039b)")
    axes[1].set_ylabel("Im(\u039b)")
    axes[1].set_title("Eigenvalues in the complex plane\n"
                      "radius = persistence, angle = oscillation (rad/TR)", fontsize=10)

    axes[2].scatter(freq, mag, alpha=0.6, color="steelblue")
    axes[2].axvspan(*RESTING_BAND, color="green", alpha=0.12,
                    label="resting band 0.01\u20130.10 Hz")
    axes[2].axvline(NYQUIST_HZ, color="black", ls=":", lw=1,
                    label=f"Nyquist {NYQUIST_HZ:.2f} Hz")
    axes[2].set_xlabel("Frequency (Hz)")
    axes[2].set_ylabel("|\u039b| (persistence)")
    axes[2].set_title("Persistence vs frequency\n"
                      "(top-left = slow, persistent, in-band = signal)", fontsize=10)
    axes[2].legend(fontsize=8)

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"Saved {out_path}")
    plt.close(fig)


def plot_mode_maps(
    W_bar_x: np.ndarray, 
    Lambda: np.ndarray,
    roi_names, 
    out_path: Path, 
    top_k: int = M,                     
    resting_band: tuple = (0.01, 0.1),    # Added to avoid global scope crashes
):
    """
    Heatmap: modes (rows) x ROIs (columns), color = |loading|.
    All selected M modes shown individually (no averaging over H).
    ROIs grouped by network. Resting band modes highlighted via text color.
    """
    NETWORKS = {
        "Basal Ganglia": [
            "lh_Ca", "lh_GPe", "lh_GPi", "lh_Pu", "lh_STH",
            "rh_Ca", "rh_GPe", "rh_GPi", "rh_Pu", "rh_STH",
        ],
        "Cerebellum": [
            "lh_cerebellum_dentate", "lh_cerebellum_motor",
            "rh_cerebellum_dentate", "rh_cerebellum_motor",
        ],
        "Cortex": [
            "lh_paracentral_smooth3mm", "lh_postcentral_smooth3mm",
            "lh_precentral_smooth3mm", "lh_superiorfrontal_smooth3mm",
            "rh_paracentral_smooth3mm", "rh_postcentral_smooth3mm",
            "rh_precentral_smooth3mm", "rh_superiorfrontal_smooth3mm",
        ],
        "TUS Targets": [
            "lh_vim",
            "lh_zi",
        ],
    }
    NETWORK_COLORS = {
        "Basal Ganglia": "gold",
        "Cerebellum":    "lightblue",
        "Cortex":        "lightgreen",
        "TUS Targets":   "salmon",
    }

    roi_names_list = list(roi_names)
    n_rois = len(roi_names_list)

    # Build ROI order grouped by network
    roi_order = []
    roi_network_labels = []
    for net_name, net_rois in NETWORKS.items():
        for r in net_rois:
            if r in roi_names_list:
                roi_order.append(roi_names_list.index(r))
                roi_network_labels.append(net_name)
                
    # Append any ROIs not in any network group at the end
    ungrouped = [i for i in range(n_rois) if i not in roi_order]
    for i in ungrouped:
        roi_order.append(i)
        roi_network_labels.append("Other")

    W_abs = np.abs(W_bar_x)
    loading = W_abs.T  # (M, n_rois)

    # Sort modes by eigenvalue magnitude (persistence)
    u = eig_units(Lambda)
    mode_order = np.argsort(u["mag"])[::-1]
    
    mode_order = mode_order[:top_k]
        
    M_display = len(mode_order)
    loading_sorted = loading[mode_order, :]
    loading_grouped = loading_sorted[:, roi_order]

    fig, ax = plt.subplots(figsize=(14, max(6, M_display * 0.22)))

    im = ax.imshow(loading_grouped, aspect="auto", cmap="hot", interpolation="nearest")
    plt.colorbar(im, ax=ax, fraction=0.02, pad=0.03, label="|loading|")

    # ROI labels on x-axis grouped by network
    ordered_roi_names = [roi_names_list[i] for i in roi_order]
    ax.set_xticks(range(len(roi_order)))
    ax.set_xticklabels(ordered_roi_names, rotation=90, fontsize=8)
    ax.set_xlabel("Brain Region (grouped by network)", fontsize=10)

    # Mode labels on y-axis
    mode_labels = [
        f"M{mode_order[i]}  |λ|={u['mag'][mode_order[i]]:.2f}  {u['freq_hz'][mode_order[i]]:.3f}Hz"
        for i in range(M_display)
    ]
    ax.set_yticks(range(M_display))
    ax.set_yticklabels(mode_labels, fontsize=7)
    ax.set_ylabel("Koopman Mode (sorted by persistence)", fontsize=10)

    # Fixed: Safe text placement using blended transform to prevent clipping above plot
    boundary = 0
    for net_name, net_rois in NETWORKS.items():
        n_in_network = sum(1 for r in net_rois if r in roi_names_list)
        if n_in_network == 0:
            continue
        if boundary > 0:
            ax.axvline(boundary - 0.5, color="white", lw=1.5, ls="--")
            
        # Transform ensures y position (1.02) is dynamically scaled just above plot bounds
        ax.text(boundary + n_in_network / 2 - 0.5, 1.02,
                net_name, ha="center", va="bottom",
                fontsize=9, fontweight="bold",
                color=NETWORK_COLORS.get(net_name, "black"),
                transform=ax.get_xaxis_transform())
        boundary += n_in_network

    # Highlight resting-band frequencies directly on the text labels
    freq_ordered = u["freq_hz"][mode_order]
    yticklabels = ax.get_yticklabels()
    
    for idx, freq in enumerate(freq_ordered):
        if resting_band[0] <= freq <= resting_band[1]:
            yticklabels[idx].set_color("darkcyan")
            yticklabels[idx].set_weight("bold")

    ax.set_title(
        "Koopman Mode Spatial Loadings\n"
        f"rows = top {M_display} modes sorted by persistence  |  columns = brain regions grouped by network",
        fontsize=11, pad=25  # Added padding to give network labels room to breathe
    )

    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


def compute_block_norms(K: np.ndarray, n_rois: int, h: int) -> np.ndarray:
    """
    Partition K (M x M, ROI-major) into n_rois x n_rois blocks of H x H.
    B[i, j] = Frobenius norm of block (i, j) = directed one-step latent
    influence of ROI j (source, t) on ROI i (target, t+1).
    """
    B = np.zeros((n_rois, n_rois))
    for i in range(n_rois):
        for j in range(n_rois):
            B[i, j] = np.linalg.norm(K[i*h:(i+1)*h, j*h:(j+1)*h])
    return B


def plot_block_coupling(B: np.ndarray, roi_names, out_path: Path):
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(B, cmap="magma", aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="block coupling norm")
    ax.set_xticks(range(len(roi_names)))
    ax.set_yticks(range(len(roi_names)))
    ax.set_xticklabels(roi_names, rotation=90, fontsize=7)
    ax.set_yticklabels(roi_names, fontsize=7)
    ax.set_xlabel("source ROI (t)")
    ax.set_ylabel("target ROI (t+1)")
    ax.set_title("Region-to-region latent coupling |K| blocks\n"
                 "(diagonal = within-region; one-step prediction, NOT causation)",
                 fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# 3. C (PER TARGET, INFERENTIAL)
# ================================================================================
 
def extract_C_diagonals(model: BRICK, target: str):
    """
    Return (subject_ids, pre, post): (n_subjects, M) raw diag(C) arrays.
    Raw g-space coordinates -- NOT ROI-labeled (see module docstring). One
    target only, no pooling across targets.
    """
    subjects = load_all()
    sids, pre, post = [], [], []
    with torch.no_grad():
        for s in subjects:
            if s["target"] != target:
                continue
            x_pre  = znorm(torch.tensor(s["mpre"],  dtype=torch.float32))
            x_post = znorm(torch.tensor(s["mpost"], dtype=torch.float32))
            sids.append(s["subject_id"])
            pre.append(np.real(model(x_pre)["C"].diag().cpu().numpy()))
            post.append(np.real(model(x_post)["C"].diag().cpu().numpy()))
    return sids, np.asarray(pre), np.asarray(post)
 
 
def paired_tests_per_coordinate(pre, post, alpha=0.05) -> pd.DataFrame:
    """
    Per g-space coordinate: paired t-test pre vs post across subjects, plus
    Wilcoxon signed-rank as a non-parametric companion (N is small, normality
    is unverifiable). FDR-corrected across coordinates. The subject is the
    unit of replication; coordinate m is the same latent direction for every
    subject because the basis is global, which is what licenses averaging
    across people.
 
    NO roi_name/roi_index/channel columns: diag(C)'s raw coordinates carry no
    ROI meaning (see module docstring). Use paired_tests_per_roi for spatial
    localization.
    """
    N, Mc = pre.shape
    diff = post - pre
    t = np.zeros(Mc); p = np.ones(Mc); w = np.ones(Mc)
    for m in range(Mc):
        if np.allclose(diff[:, m], 0.0):
            t[m], p[m] = 0.0, 1.0
        else:
            t[m], p[m] = stats.ttest_rel(pre[:, m], post[:, m])
        try:
            w[m] = stats.wilcoxon(pre[:, m], post[:, m]).pvalue
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
    Per ROI, on DECODER-PROJECTED values (see project_to_roi /
    compute_roi_projection_weights), NOT a coordinate-index reshape. Same
    paired t-test + Wilcoxon + FDR structure as the coordinate-level test,
    but now the ROI label is actually valid: pre_roi/post_roi were produced
    by pushing diag(C) through the decoder's own weights.
    """
    N, n_rois = pre_roi.shape
    diff = post_roi - pre_roi
    t = np.zeros(n_rois); p = np.ones(n_rois); w = np.ones(n_rois)
    for i in range(n_rois):
        if np.allclose(diff[:, i], 0.0):
            t[i], p[i] = 0.0, 1.0
        else:
            t[i], p[i] = stats.ttest_rel(pre_roi[:, i], post_roi[:, i])
        try:
            w[i] = stats.wilcoxon(pre_roi[:, i], post_roi[:, i]).pvalue
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
    """
    Omnibus: paired t-test on ||C||_F. Since C is diagonal,
    ||C||_F = sqrt(sum diag^2), a coarse single-number summary of the gains.
    """
    pre_norm  = np.sqrt((pre ** 2).sum(1))
    post_norm = np.sqrt((post ** 2).sum(1))
    t, p = stats.ttest_rel(pre_norm, post_norm)
    try:
        w = stats.wilcoxon(pre_norm, post_norm).pvalue
    except ValueError:
        w = 1.0
    return float(t), float(p), float(w)
 
 
def plot_delta_C_roi(roi_delta: np.ndarray, roi_names, out_path: Path, target: str):
    """
    Per-ROI mean paired difference (post - pre), on decoder-projected values.
    A single length-N_ROIS bar -- no (ROI x channel) grid, since C's raw
    coordinates have no channel-to-ROI meaning to grid on.
    """
    n_rois = len(roi_names)
    fig, ax = plt.subplots(figsize=(8, 0.35 * n_rois + 1))
    colors = ["crimson" if v > 0 else "steelblue" for v in roi_delta]
    ax.barh(range(n_rois), roi_delta, color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(range(n_rois))
    ax.set_yticklabels(list(roi_names), fontsize=8)
    ax.invert_yaxis()   # keep first ROI at top, matches other ROI-axis plots
    ax.set_xlabel("\u0394 control gain (post - pre), decoder-projected")
    ax.set_title(f"Per-ROI \u0394C \u2014 {target.upper()}", fontsize=11)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)
 
 
def plot_delta_C_coordinates(coord_delta: np.ndarray, out_path: Path, target: str):
    """
    Raw coordinate-level mean paired difference, unlabeled (no ROI axis --
    these coordinates carry no ROI meaning). Useful only to see the overall
    distribution/shape of the effect across g-space.
    """
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.bar(range(len(coord_delta)), coord_delta, color="slategray")
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xlabel("g-space coordinate index (not ROI-labeled)")
    ax.set_ylabel("\u0394 diag(C)")
    ax.set_title(f"Coordinate-level \u0394C \u2014 {target.upper()} "
                "(raw, no spatial meaning; see per-ROI plot for anatomy)",
                fontsize=10)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Saved {out_path}")
    plt.close(fig)


# ================================================================================
# 4. MAIN
# ================================================================================

def run_K_descriptives(model, top_k):
    print("\n--- K (global, descriptive) ---")
    K, Lambda, W_bar_x = compute_K(model)

    eig_df = eigenvalue_table(Lambda)
    eig_path = RESULTS_DIR / "koopman_eigenvalues.csv"
    eig_df.to_csv(eig_path, index=False)
    print(f"Saved {eig_path}")
    print(f"  Persistent modes (|\u039b|>0.9): {(np.abs(Lambda) > 0.9).sum()} / {len(Lambda)}")

    plot_spectrum(Lambda, FIGURES_DIR / "koopman_spectrum.png")
    plot_mode_maps(W_bar_x, Lambda, TARGET_ROIS,
                   FIGURES_DIR / "koopman_mode_maps.png", top_k=top_k)
    B = compute_block_norms(K, N_ROIS, H)
    plot_block_coupling(B, TARGET_ROIS, FIGURES_DIR / "K_region_coupling.png")


def run_C_inference(model, W_bar_x, target, alpha):
    print(f"\n--- C inference: target = {target.upper()} ---")
    sids, pre, post = extract_C_diagonals(model, target)
    n = len(sids)
    print(f"  {n} subjects")
    if n < 2:
        print(f"  Skipping {target}: need >=2 subjects for a paired test.")
        return
 
    # Coordinate-level: valid stats, no ROI label.
    coord_df = paired_tests_per_coordinate(pre, post, alpha)
 
    # ROI-level: decoder-projected, this is the spatially valid one.
    roi_weights = compute_roi_projection_weights(W_bar_x)
    pre_roi  = project_to_roi(pre,  roi_weights)
    post_roi = project_to_roi(post, roi_weights)
    roi_df = paired_tests_per_roi(pre_roi, post_roi, TARGET_ROIS, alpha)
 
    t_n, p_n, w_n = norm_omnibus(pre, post)
 
    coord_path = RESULTS_DIR / f"statistical_results_coord_{target}.csv"
    roi_path   = RESULTS_DIR / f"statistical_results_roi_{target}.csv"
    coord_df.to_csv(coord_path, index=False)
    roi_df.to_csv(roi_path, index=False)
    print(f"  Saved {coord_path}")
    print(f"  Saved {roi_path}")
 
    n_sig_coord = int(coord_df["significant"].sum())
    n_sig_roi   = int(roi_df["significant"].sum())
    print(f"  Coordinates surviving FDR (alpha={alpha}): {n_sig_coord} / {len(coord_df)}")
    print(f"  ROIs surviving FDR (alpha={alpha}): {n_sig_roi} / {len(roi_df)}")
    print(f"  ||C||_F omnibus: t={t_n:.3f}, p={p_n:.4f}, wilcoxon p={w_n:.4f}")
 
    if n_sig_roi:
        sig = roi_df[roi_df["significant"]].sort_values("p_value_fdr")
        print("  Significant ROIs (decoder-projected):")
        for _, r in sig.iterrows():
            flag = "" if r["wilcoxon_p"] < alpha else "  [Wilcoxon disagrees]"
            print(f"    {r['roi_name']:<14} \u0394={r['delta']:+.4f}  "
                  f"p_fdr={r['p_value_fdr']:.4f}{flag}")
 
    plot_delta_C_roi(roi_df["delta"].values, TARGET_ROIS,
                     FIGURES_DIR / f"delta_C_roi_{target}.png", target)
    plot_delta_C_coordinates(coord_df["delta"].values,
                             FIGURES_DIR / f"delta_C_coord_{target}.png", target)


def main(targets, alpha=0.05, top_k=M):
    print("=" * 64)
    print("BRICK Pre/Post Analysis")
    print("=" * 64)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
 
    subjects = load_all()
    verify_roi_consistency(subjects)
 
    model = load_model()
    K, Lambda, W_bar_x = compute_K(model)   # kept for run_C_inference's W_bar_x

    run_K_descriptives(model, top_k=top_k)   # recomputes K/Lambda/W_bar_x internally — cheap, no grad
    for target in targets:
        run_C_inference(model, W_bar_x, target, alpha)


# ================================================================================
# ENTRY POINT
# ================================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=str, default=None, choices=["vim", "zi"],
                        help="Run one target. Default: both, separately (never pooled).")
    parser.add_argument("--alpha", type=float, default=0.05,
                        help="FDR significance threshold.")
    parser.add_argument("--top-k", type=int, default=M,
                        help="Number of most-persistent modes to plot.")
    args = parser.parse_args()

    targets = [args.target] if args.target else ["vim", "zi"]
    main(targets, alpha=args.alpha, top_k=args.top_k)