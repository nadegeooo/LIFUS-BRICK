import os
import glob
import pytest
from pathlib import Path
import numpy as np
import scipy.io

# --- SETUP ---
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
PREPROCESSED_DIR = ROOT_DIR / "data" / "preprocessed_data"

# --- CONSTANTS ---
N_TIMEPOINTS = 240
N_ROIS = 24
VIM_IDX = 11
ZI_IDX = 12

# --- EXPECTED SUBJECTS ---
EXPECTED_SUBJECTS = [
    "sub-fuspd01", "sub-fuspd03", "sub-fuspd04", "sub-fuspd06", "sub-fuspd07",
    "sub-fuspd08", "sub-fuspd09", "sub-fuspd11", "sub-fuspd12", "sub-fuspd13",
    "sub-fuspd15", "sub-fuspd16", "sub-fuspd18", "sub-fuspd19", "sub-fuspd20",
    "sub-fuspd22", "sub-fuspd23", "sub-fuspd24", "sub-fuspd26"
]

# --- FILE DISCOVERY ---
VIM_FILES = []
ZI_FILES = []
for sub in EXPECTED_SUBJECTS:
    VIM_FILES.extend(glob.glob(str(PREPROCESSED_DIR / f"{sub}_*_roi-vim_meants.mat")))
    ZI_FILES.extend(glob.glob(str(PREPROCESSED_DIR / f"{sub}_*_roi-zi_meants.mat")))

ALL_FILES = VIM_FILES + ZI_FILES

if not ALL_FILES:
    pytest.skip("No preprocessed files found — run preprocessing first", allow_module_level=True)


# --- FIXTURES ---
@pytest.fixture(scope="session")
def loaded_mats():
    """Load all .mat files once per session and cache them."""
    return {f: scipy.io.loadmat(f) for f in ALL_FILES}


# ================================================================================
# 1. FILE COMPLETENESS TESTS
# ================================================================================
@pytest.mark.parametrize("subject", EXPECTED_SUBJECTS)
def test_files_exist(subject):
    """Ensure every subject has exactly one preprocessed VIM and one ZI file."""
    vim_match = glob.glob(str(PREPROCESSED_DIR / f"{subject}_*_roi-vim_meants.mat"))
    zi_match = glob.glob(str(PREPROCESSED_DIR / f"{subject}_*_roi-zi_meants.mat"))

    assert len(vim_match) == 1, f"Missing or duplicate 'roi-vim' file for {subject}."
    assert len(zi_match) == 1, f"Missing or duplicate 'roi-zi' file for {subject}."


# ================================================================================
# 2. MATRIX SHAPE & CONTENT TESTS
# ================================================================================
@pytest.mark.parametrize("filepath", ALL_FILES)
def test_matrix_structures(filepath, loaded_mats):
    """Verify keys exist, shapes are (N_TIMEPOINTS, N_ROIS), and pre/post match."""
    fname = os.path.basename(filepath)
    mat = loaded_mats[filepath]

    # Keys must exist before any shape checks
    assert 'mpre' in mat, f"{fname} is missing key 'mpre'"
    assert 'mpost' in mat, f"{fname} is missing key 'mpost'"

    assert mat['mpre'].shape == (N_TIMEPOINTS, N_ROIS), \
        f"{fname} 'mpre' shape is {mat['mpre'].shape}, expected ({N_TIMEPOINTS}, {N_ROIS})"
    assert mat['mpost'].shape == (N_TIMEPOINTS, N_ROIS), \
        f"{fname} 'mpost' shape is {mat['mpost'].shape}, expected ({N_TIMEPOINTS}, {N_ROIS})"

    assert mat['mpre'].shape == mat['mpost'].shape, \
        f"{fname} pre and post have different shapes: {mat['mpre'].shape} vs {mat['mpost'].shape}"


@pytest.mark.parametrize("filepath", ALL_FILES)
def test_no_nan_or_inf(filepath, loaded_mats):
    """Ensure no NaN or Inf values exist in either pre or post matrices."""
    fname = os.path.basename(filepath)
    mat = loaded_mats[filepath]

    assert np.isfinite(mat['mpre']).all(), f"{fname} contains NaN or Inf values in 'mpre'."
    assert np.isfinite(mat['mpost']).all(), f"{fname} contains NaN or Inf values in 'mpost'."


@pytest.mark.parametrize("filepath", ALL_FILES)
def test_columns_populated(filepath, loaded_mats):
    """Confirm VIM (index 11) and ZI (index 12) columns are not flat zero vectors."""
    fname = os.path.basename(filepath)
    mat = loaded_mats[filepath]

    assert not np.all(mat['mpre'][:, VIM_IDX] == 0), f"{fname} lh_vim column is empty zeros in mpre"
    assert not np.all(mat['mpre'][:, ZI_IDX] == 0), f"{fname} lh_zi column is empty zeros in mpre"
    assert not np.all(mat['mpost'][:, VIM_IDX] == 0), f"{fname} lh_vim column is empty zeros in mpost"
    assert not np.all(mat['mpost'][:, ZI_IDX] == 0), f"{fname} lh_zi column is empty zeros in mpost"


# ================================================================================
# 3. FUNCTIONAL CONNECTIVITY (FC) INVARIANT TESTS
# ================================================================================
@pytest.mark.parametrize("filepath", ALL_FILES)
def test_fc_invariants(filepath, loaded_mats):
    """Compute FC via Pearson correlation and enforce standard mathematical properties."""
    fname = os.path.basename(filepath)
    mat = loaded_mats[filepath]

    for condition in ['mpre', 'mpost']:
        data = mat[condition]

        # rowvar=False: rows=timepoints, columns=ROIs
        fc_matrix = np.corrcoef(data, rowvar=False)

        assert fc_matrix.shape == (N_ROIS, N_ROIS), \
            f"{fname} ({condition}) FC shape is {fc_matrix.shape}, expected ({N_ROIS}, {N_ROIS})"

        assert np.allclose(fc_matrix, fc_matrix.T, atol=1e-7), \
            f"{fname} ({condition}) FC matrix is asymmetric."

        assert np.allclose(np.diag(fc_matrix), 1.0, atol=1e-7), \
            f"{fname} ({condition}) FC matrix diagonal is not 1.0."

        assert (fc_matrix >= -1.0001).all() and (fc_matrix <= 1.0001).all(), \
            f"{fname} ({condition}) FC matrix contains values outside [-1, 1]."