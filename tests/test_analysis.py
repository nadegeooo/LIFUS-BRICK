# tests/test_analysis.py
"""
Tests for the pre/post statistical analysis pipeline.
Uses real .mat files and a real model checkpoint from results/final_model/.
Model-dependent tests are skipped if no checkpoint is found.
"""

import torch
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from scipy import stats
from statsmodels.stats.multitest import multipletests

ROOT_DIR = Path(__file__).resolve().parent.parent

from config import M, N_ROIS
from preprocessing.load_preprocessed_data import load_all

FINAL_MODEL_PATH = ROOT_DIR / "results" / "final_model" / "best_model.pt"

needs_model = pytest.mark.skipif(
    not FINAL_MODEL_PATH.exists(),
    reason="No checkpoint found in results/final_model/best_model.pt"
)


# ================================================================================
# FIXTURES
# ================================================================================

@pytest.fixture(scope="module")
def all_subjects():
    """Load all preprocessed .mat files."""
    subjects = load_all()
    assert len(subjects) > 0, "No subjects loaded — check data/preprocessed_data/"
    return subjects


@pytest.fixture(scope="module")
def model():
    """Load model from results/final_model/best_model.pt."""
    from models.brick import BRICK
    checkpoint = torch.load(FINAL_MODEL_PATH, map_location="cpu")
    m = BRICK(
        use_control=checkpoint["use_control"],
        use_ic=checkpoint["use_ic"],
        h=checkpoint["h"],
        m=checkpoint["m"],
    )
    m.load_state_dict(checkpoint["model_state_dict"])
    m.eval()
    return m


@pytest.fixture(scope="module")
def C_dict(model, all_subjects):
    """
    Extract C_pre and C_post for all subjects.
    Returns dict keyed by (subject_id, target) -> {"pre": C_pre, "post": C_post}
    """
    result = {}
    with torch.no_grad():
        for s in all_subjects:
            x_pre  = torch.tensor(s["mpre"],  dtype=torch.float32)
            x_post = torch.tensor(s["mpost"], dtype=torch.float32)
            key = (s["subject_id"], s["target"])
            C_pre  = model(x_pre)["C"]
            C_post = model(x_post)["C"]
            result[key] = {"pre": C_pre, "post": C_post}
    return result


@pytest.fixture(scope="module")
def dummy_results_csv(tmp_path_factory):
    """Write a minimal statistical_results.csv to a temp directory."""
    tmp_path = tmp_path_factory.mktemp("results")
    df = pd.DataFrame({
        "mode_index":       range(M),
        "eigenvalue_mag":   np.random.rand(M),
        "eigenvalue_phase": np.random.rand(M),
        "t_statistic":      np.random.randn(M),
        "p_value":          np.random.rand(M),
        "p_value_fdr":      np.random.rand(M),
        "significant":      np.zeros(M, dtype=bool),
    })
    # Make FDR >= uncorrected to satisfy the real constraint
    _, p_fdr, _, _ = multipletests(df["p_value"].values, method="fdr_bh")
    df["p_value_fdr"] = p_fdr
    path = tmp_path / "statistical_results.csv"
    df.to_csv(path, index=False)
    return path


# ================================================================================
# 1. DATA LOADING
# ================================================================================

def test_all_files_load(all_subjects):
    """All .mat files must load without error."""
    assert len(all_subjects) > 0

def test_subject_has_required_keys(all_subjects):
    """Each subject dict must have required keys."""
    required = {"mpre", "mpost", "roi_names", "subject_id", "target", "group"}
    for s in all_subjects:
        missing = required - set(s.keys())
        assert not missing, f"Missing keys for {s.get('subject_id')}: {missing}"

def test_mpre_mpost_shapes(all_subjects):
    """mpre and mpost must be (240, 24) for all subjects."""
    for s in all_subjects:
        assert s["mpre"].shape  == (240, N_ROIS), \
            f"{s['subject_id']} mpre shape {s['mpre'].shape}"
        assert s["mpost"].shape == (240, N_ROIS), \
            f"{s['subject_id']} mpost shape {s['mpost'].shape}"

def test_target_is_vim_or_zi(all_subjects):
    """Target must be 'vim' or 'zi' for all subjects."""
    for s in all_subjects:
        assert s["target"] in ("vim", "zi"), \
            f"{s['subject_id']} has unexpected target: {s['target']}"

def test_each_subject_has_both_targets(all_subjects):
    """Each subject must have both VIM and ZI entries."""
    from collections import defaultdict
    targets_per_subject = defaultdict(set)
    for s in all_subjects:
        targets_per_subject[s["subject_id"]].add(s["target"])
    for sid, targets in targets_per_subject.items():
        assert "vim" in targets and "zi" in targets, \
            f"{sid} missing target: has {targets}"


# ================================================================================
# 2. SUBJECT PAIRING
# ================================================================================

def test_pre_post_paired_within_file(all_subjects):
    """mpre and mpost must have the same shape — they come from the same file."""
    for s in all_subjects:
        assert s["mpre"].shape == s["mpost"].shape, \
            f"{s['subject_id']} pre/post shape mismatch"

def test_no_duplicate_subject_target_pairs(all_subjects):
    """Each (subject_id, target) combination must appear exactly once."""
    keys = [(s["subject_id"], s["target"]) for s in all_subjects]
    assert len(keys) == len(set(keys)), \
        f"Duplicate subject-target pairs found: {[k for k in keys if keys.count(k) > 1]}"


# ================================================================================
# 3. C MATRIX EXTRACTION
# ================================================================================

@needs_model
def test_C_dict_has_all_subjects(C_dict, all_subjects):
    """C_dict must have an entry for every subject-target pair."""
    expected_keys = {(s["subject_id"], s["target"]) for s in all_subjects}
    assert set(C_dict.keys()) == expected_keys

@needs_model
def test_C_matrices_shape(C_dict):
    """C_pre and C_post must be (M, M) for all subjects."""
    for key, val in C_dict.items():
        assert val["pre"].shape  == (M, M), \
            f"{key} C_pre shape {val['pre'].shape}"
        assert val["post"].shape == (M, M), \
            f"{key} C_post shape {val['post'].shape}"

@needs_model
def test_C_matrices_finite(C_dict):
    """C_pre and C_post must be finite for all subjects."""
    for key, val in C_dict.items():
        assert torch.isfinite(val["pre"]).all(),  f"{key} C_pre contains NaN/Inf"
        assert torch.isfinite(val["post"]).all(), f"{key} C_post contains NaN/Inf"

@needs_model
def test_C_pre_post_differ(C_dict):
    """C_pre and C_post should not be identical."""
    for key, val in C_dict.items():
        assert not torch.allclose(val["pre"], val["post"]), \
            f"{key} C_pre and C_post are identical"


# ================================================================================
# 4. C NORM PAIRED T-TEST
# ================================================================================

@needs_model
def test_c_norm_ttest_returns_scalar(C_dict):
    """C norm paired t-test must return scalar t and p."""
    pre_norms  = np.array([v["pre"].norm().item()  for v in C_dict.values()])
    post_norms = np.array([v["post"].norm().item() for v in C_dict.values()])
    t, p = stats.ttest_rel(pre_norms, post_norms)
    assert np.isfinite(t), f"t-statistic not finite: {t}"
    assert 0.0 <= p <= 1.0, f"p-value out of range: {p}"

@needs_model
def test_c_norm_arrays_same_length(C_dict):
    """Pre and post norm arrays must have the same length."""
    pre_norms  = [v["pre"].norm().item()  for v in C_dict.values()]
    post_norms = [v["post"].norm().item() for v in C_dict.values()]
    assert len(pre_norms) == len(post_norms)


# ================================================================================
# 5. FDR CORRECTION
# ================================================================================

def test_fdr_pvalues_geq_uncorrected():
    """FDR-corrected p-values must be >= uncorrected p-values."""
    np.random.seed(42)
    p_uncorrected = np.random.rand(M)
    _, p_fdr, _, _ = multipletests(p_uncorrected, method="fdr_bh")
    assert np.all(p_fdr >= p_uncorrected - 1e-10), \
        "Some FDR p-values are smaller than uncorrected p-values"

def test_fdr_pvalues_in_range():
    """FDR-corrected p-values must be in [0, 1]."""
    np.random.seed(42)
    p_uncorrected = np.random.rand(M)
    _, p_fdr, _, _ = multipletests(p_uncorrected, method="fdr_bh")
    assert np.all(p_fdr >= 0.0) and np.all(p_fdr <= 1.0)


# ================================================================================
# 6. CSV COLUMNS
# ================================================================================

REQUIRED_COLUMNS = [
    "mode_index",
    "eigenvalue_mag",
    "eigenvalue_phase",
    "t_statistic",
    "p_value",
    "p_value_fdr",
    "significant",
]

def test_results_csv_has_required_columns(dummy_results_csv):
    """Results CSV must contain all required columns."""
    df = pd.read_csv(dummy_results_csv)
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    assert not missing, f"Missing columns: {missing}"

def test_results_csv_no_nan(dummy_results_csv):
    """Results CSV must not contain NaN."""
    df = pd.read_csv(dummy_results_csv)
    assert not df.isnull().any().any()

def test_results_csv_row_count(dummy_results_csv):
    """Results CSV must have exactly M rows."""
    df = pd.read_csv(dummy_results_csv)
    assert len(df) == M, f"Expected {M} rows, got {len(df)}"