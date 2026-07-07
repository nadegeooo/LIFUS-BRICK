# tests/test_dcm_baseline.py
"""
Tests for the VAR baseline analysis pipeline.
Write these before implementing analysis/dcm_baseline.py.
Tests should initially fail and pass once the implementation is complete.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent.parent

from preprocessing.load_preprocessed_data import load_all, TARGET_ROIS
from config import N_ROIS
from analysis.dcm_baseline import fit_var, predict_mse, get_outgoing, get_incoming, run_var_ttest, build_comparison_table

# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture(scope="module")
def all_subjects():
    """Load all preprocessed .mat files."""
    return load_all()


@pytest.fixture(scope="module")
def dummy_timeseries():
    """Random BOLD timeseries for a single subject, shape (240, 24)."""
    np.random.seed(42)
    return np.random.randn(240, N_ROIS)


@pytest.fixture(scope="module")
def fitted_var(dummy_timeseries):
    """Fit VAR on dummy data and return A_eff."""
    return fit_var(dummy_timeseries)


@pytest.fixture(scope="module")
def all_var_results(all_subjects):
    """
    Fit VAR on all real subjects, return list of dicts with
    A_pre and A_post per subject-target pair.
    """
    results = []
    for s in all_subjects:
        results.append({
            "subject_id": s["subject_id"],
            "target":     s["target"],
            "A_pre":      fit_var(s["mpre"]),
            "A_post":     fit_var(s["mpost"]),
        })
    return results


# ================================================================================
# 1. VAR FITTING
# ================================================================================

def test_var_shape(fitted_var):
    """VAR coefficient matrix must be (N_ROIS, N_ROIS)."""
    assert fitted_var.shape == (N_ROIS, N_ROIS), \
        f"Expected ({N_ROIS}, {N_ROIS}), got {fitted_var.shape}"


def test_var_finite(fitted_var):
    """VAR coefficient matrix must be finite."""
    assert np.isfinite(fitted_var).all(), \
        "VAR matrix contains NaN or Inf"


def test_var_better_than_zero(dummy_timeseries):
    """VAR one-step prediction MSE must be lower than predicting with zero matrix."""
    A_eff = fit_var(dummy_timeseries)
    mse_var  = predict_mse(A_eff, dummy_timeseries)
    mse_zero = predict_mse(np.zeros((N_ROIS, N_ROIS)), dummy_timeseries)
    assert mse_var < mse_zero, \
        f"VAR MSE ({mse_var:.4f}) is not better than zero matrix ({mse_zero:.4f})"


def test_var_real(fitted_var):
    """VAR matrix must be real-valued."""
    assert fitted_var.dtype in (np.float32, np.float64), \
        f"VAR matrix dtype is {fitted_var.dtype}, expected float"


# ================================================================================
# 2. CONNECTION EXTRACTION
# ================================================================================

def test_outgoing_connections_shape(fitted_var):
    """Outgoing connections (vim drives others) = column, shape (N_ROIS,)."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    outgoing = get_outgoing(fitted_var, roi_idx)
    assert outgoing.shape == (N_ROIS,), \
        f"Expected ({N_ROIS},), got {outgoing.shape}"


def test_incoming_connections_shape(fitted_var):
    """Incoming connections (others drive vim) = row, shape (N_ROIS,)."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    incoming = get_incoming(fitted_var, roi_idx)
    assert incoming.shape == (N_ROIS,), \
        f"Expected ({N_ROIS},), got {incoming.shape}"


def test_outgoing_is_row(fitted_var):
    """Outgoing connections from ROI = column of A_eff (vim drives others)."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    outgoing = get_outgoing(fitted_var, roi_idx)
    assert np.allclose(outgoing, fitted_var[:, roi_idx]), \
        "Outgoing connections do not match the ROI column of A_eff"


def test_incoming_is_column(fitted_var):
    """Incoming connections to ROI = row of A_eff (others drive vim)."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    incoming = get_incoming(fitted_var, roi_idx)
    assert np.allclose(incoming, fitted_var[roi_idx, :]), \
        "Incoming connections do not match the ROI row of A_eff"

# ================================================================================
# 3. PAIRED T-TESTS
# ================================================================================

def test_ttest_pvalue_array_length(all_var_results):
    """Paired t-test p-values must have length N_ROIS for outgoing connections."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    t_stats, p_vals = run_var_ttest(all_var_results, roi_idx, connection="outgoing")
    assert len(p_vals) == N_ROIS, \
        f"Expected {N_ROIS} p-values, got {len(p_vals)}"


def test_ttest_pvalues_in_range(all_var_results):
    """All p-values must be in [0, 1]."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    _, p_vals = run_var_ttest(all_var_results, roi_idx, connection="outgoing")
    assert np.all(p_vals >= 0.0) and np.all(p_vals <= 1.0), \
        "p-values out of [0, 1] range"


def test_ttest_tstat_array_length(all_var_results):
    """T-statistics must have length N_ROIS."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    t_stats, _ = run_var_ttest(all_var_results, roi_idx, connection="outgoing")
    assert len(t_stats) == N_ROIS, \
        f"Expected {N_ROIS} t-statistics, got {len(t_stats)}"


def test_ttest_finite(all_var_results):
    """T-statistics and p-values must be finite."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    t_stats, p_vals = run_var_ttest(all_var_results, roi_idx, connection="outgoing")
    assert np.isfinite(t_stats).all(), "t-statistics contain NaN or Inf"
    assert np.isfinite(p_vals).all(),  "p-values contain NaN or Inf"


# ================================================================================
# 4. FDR CORRECTION
# ================================================================================

def test_fdr_pvalues_geq_uncorrected(all_var_results):
    """FDR-corrected p-values must be >= uncorrected p-values."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    _, p_vals = run_var_ttest(all_var_results, roi_idx, connection="outgoing")
    _, p_fdr, _, _ = multipletests(p_vals, method="fdr_bh")
    assert np.all(p_fdr >= p_vals - 1e-10), \
        "Some FDR p-values are smaller than uncorrected p-values"


# ================================================================================
# 5. COMPARISON TABLE
# ================================================================================

REQUIRED_TABLE_COLUMNS = [
    "method",
    "n_significant",
    "median_cohens_d",
    "interpretation",
]

def test_comparison_table_produced(all_var_results):
    """Comparison table must be produced without errors and have required columns."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    table = build_comparison_table(all_var_results, roi_idx)
    assert isinstance(table, pd.DataFrame), "Comparison table must be a DataFrame"
    missing = [col for col in REQUIRED_TABLE_COLUMNS if col not in table.columns]
    assert not missing, f"Missing columns in comparison table: {missing}"


def test_comparison_table_has_two_rows(all_var_results):
    """Comparison table must have exactly 2 rows (BRICK and VAR)."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    table = build_comparison_table(all_var_results, roi_idx)
    assert len(table) == 2, f"Expected 2 rows, got {len(table)}"


def test_comparison_table_methods(all_var_results):
    """Comparison table must have one BRICK row and one VAR row."""
    roi_idx = TARGET_ROIS.index("lh_vim")
    table = build_comparison_table(all_var_results, roi_idx)
    methods = set(table["method"].str.upper())
    assert "BRICK" in methods, "Missing BRICK row in comparison table"
    assert "VAR"   in methods, "Missing VAR row in comparison table"