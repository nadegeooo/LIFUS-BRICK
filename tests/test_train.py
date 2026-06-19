"""
================================================================================
Tests for BRICKDataset and training loop
================================================================================

TDD: these tests are written before the implementation.
Run with: pytest tests/test_train.py -v

All tests that require actual .mat files are marked with:
    @pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
so the suite can run in CI without data.
"""

import csv
import json
import pytest
import torch
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data" / "preprocessed_data"
RESULTS_DIR = ROOT_DIR / "results" / "training"

DATA_AVAILABLE = DATA_DIR.exists() and any(DATA_DIR.glob("*.mat"))

from config import N_ROIS, T as T_DATA


# ================================================================================
# 1. DATASET LENGTH
# ================================================================================
@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_dataset_length():
    """
    Dataset must contain n_files * 2 items (one pre, one post per file).
    With 38 .mat files (19 subjects x 2 targets): 38 * 2 = 76 items.
    """
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    n_files = len(list(DATA_DIR.glob("*.mat")))
    assert len(ds) == n_files * 2, \
        f"Expected {n_files * 2} items, got {len(ds)}"


# ================================================================================
# 2. ITEM STRUCTURE AND SHAPES
# ================================================================================
@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_item_returns_correct_keys():
    """Each item must contain x, fc, lifus_condition, first_target and metadata."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    item = ds[0]
    required_keys = {
        "x", "fc", "lifus_condition", "first_target",
        "subject_id", "target", "condition_str", "group_str"
    }
    assert required_keys == set(item.keys()), \
        f"Missing keys: {required_keys - set(item.keys())}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_x_shape():
    """x must be (T, N)."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    item = ds[0]
    assert item["x"].shape == (T_DATA, N_ROIS), \
        f"Expected ({T_DATA}, {N_ROIS}), got {item['x'].shape}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_fc_shape():
    """FC matrix must be (N, N)."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    item = ds[0]
    assert item["fc"].shape == (N_ROIS, N_ROIS), \
        f"Expected ({N_ROIS}, {N_ROIS}), got {item['fc'].shape}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_fc_is_valid_correlation_matrix():
    """FC must be symmetric with diagonal 1.0 and values in [-1, 1]."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    fc = ds[0]["fc"]
    assert torch.allclose(fc, fc.T, atol=1e-5), "FC is not symmetric"
    assert torch.allclose(torch.diag(fc), torch.ones(N_ROIS), atol=1e-5), \
        "FC diagonal is not 1.0"
    assert (fc >= -1.0 - 1e-5).all() and (fc <= 1.0 + 1e-5).all(), \
        "FC values outside [-1, 1]"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_lifus_condition_is_binary():
    """lifus_condition must be 0 (pre) or 1 (post)."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    conditions = set(ds[i]["lifus_condition"].item() for i in range(len(ds)))
    assert conditions == {0, 1}, \
        f"Expected {{0, 1}}, got {conditions}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_first_target_is_binary():
    """first_target must be 0 (VIM_first) or 1 (ZI_first)."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    targets = set(ds[i]["first_target"].item() for i in range(len(ds)))
    assert targets == {0, 1}, \
        f"Expected {{0, 1}}, got {targets}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_condition_str_values():
    """condition_str must be 'mpre' or 'mpost'."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    strs = set(ds[i]["condition_str"] for i in range(len(ds)))
    assert strs == {"mpre", "mpost"}, \
        f"Expected {{'mpre', 'mpost'}}, got {strs}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_group_str_values():
    """group_str must be 'VIM_first' or 'ZI_first'."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    strs = set(ds[i]["group_str"] for i in range(len(ds)))
    assert strs == {"VIM_first", "ZI_first"}, \
        f"Expected {{'VIM_first', 'ZI_first'}}, got {strs}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_x_is_float32():
    """x must be float32 for model compatibility."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    assert ds[0]["x"].dtype == torch.float32, \
        f"Expected float32, got {ds[0]['x'].dtype}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_x_no_nan_inf():
    """x must not contain NaN or Inf."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    for i in range(len(ds)):
        assert torch.isfinite(ds[i]["x"]).all(), \
            f"NaN or Inf in x at index {i}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_each_file_contributes_pre_and_post():
    """Each .mat file must contribute exactly one pre and one post item."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    from collections import Counter
    # Group by (subject_id, target) pair
    pairs = Counter(
        (ds[i]["subject_id"], ds[i]["target"], ds[i]["condition_str"])
        for i in range(len(ds))
    )
    # Each (subject, target, condition) should appear exactly once
    assert all(v == 1 for v in pairs.values()), \
        f"Duplicate items found: {[k for k, v in pairs.items() if v > 1]}"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_both_groups_present():
    """Dataset must contain both VIM_first and ZI_first subjects."""
    from training.dataset import BRICKDataset
    ds = BRICKDataset(DATA_DIR)
    
    groups = set(ds[i]["group_str"] for i in [0, 1, 2, len(ds)-3, len(ds)-2, len(ds)-1])
    first_targets = set(ds[i]["first_target"].item() for i in [0, 1, 2, len(ds)-3, len(ds)-2, len(ds)-1])
    
    print(f"\nFirst 3: {[ds[i]['group_str'] for i in range(3)]}")
    print(f"Last 3:  {[ds[i]['group_str'] for i in range(len(ds)-3, len(ds))]}")
    
    assert {"VIM_first", "ZI_first"} == groups or \
           {"VIM_first", "ZI_first"} == set(ds[i]["group_str"] for i in range(len(ds))), \
        f"Not both groups present. Found: {set(ds[i]['group_str'] for i in range(len(ds)))}"


# ================================================================================
# 3. SUBJECT SPLIT — NO LEAKAGE
# ================================================================================
@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_subject_ids_disjoint_across_splits():
    """
    Subject IDs must be completely disjoint across train, val, and test splits.
    No subject should appear in more than one split.
    """
    from training.dataset import BRICKDataset, split_dataset
    ds = BRICKDataset(DATA_DIR)
    train_ds, val_ds, test_ds = split_dataset(ds)

    train_ids = set(train_ds.dataset[i]["subject_id"] for i in train_ds.indices)
    val_ids   = set(val_ds.dataset[i]["subject_id"]   for i in val_ds.indices)
    test_ids  = set(test_ds.dataset[i]["subject_id"]  for i in test_ds.indices)

    assert train_ids.isdisjoint(val_ids), \
        f"Train/val overlap: {train_ids & val_ids}"
    assert train_ids.isdisjoint(test_ids), \
        f"Train/test overlap: {train_ids & test_ids}"
    assert val_ids.isdisjoint(test_ids), \
        f"Val/test overlap: {val_ids & test_ids}"


# ================================================================================
# 4. SPLIT PROPORTIONS
# ================================================================================
@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_split_proportions():
    """
    Split proportions must be approximately 70/10/20.
    Tolerance of ±10% accounts for rounding with only 19 subjects.
    """
    from training.dataset import BRICKDataset, split_dataset
    ds = BRICKDataset(DATA_DIR)
    train_ds, val_ds, test_ds = split_dataset(ds)

    total = len(ds)
    train_pct = len(train_ds) / total
    val_pct   = len(val_ds)   / total
    test_pct  = len(test_ds)  / total

    assert 0.60 <= train_pct <= 0.80, \
        f"Train proportion {train_pct:.2f} outside [0.60, 0.80]"
    assert 0.05 <= val_pct   <= 0.20, \
        f"Val proportion {val_pct:.2f} outside [0.05, 0.20]"
    assert 0.15 <= test_pct  <= 0.30, \
        f"Test proportion {test_pct:.2f} outside [0.15, 0.30]"


@pytest.mark.skipif(not DATA_AVAILABLE, reason="preprocessed data not found")
def test_all_items_assigned_to_exactly_one_split():
    """Every item must appear in exactly one split — no items lost or duplicated."""
    from training.dataset import BRICKDataset, split_dataset
    ds = BRICKDataset(DATA_DIR)
    train_ds, val_ds, test_ds = split_dataset(ds)

    all_indices = set(train_ds.indices) | set(val_ds.indices) | set(test_ds.indices)
    assert len(all_indices) == len(ds), \
        f"Expected {len(ds)} unique indices, got {len(all_indices)}"
    assert len(train_ds.indices) + len(val_ds.indices) + len(test_ds.indices) == len(ds), \
        "Items appear in multiple splits"


# ================================================================================
# 5. LOSS HISTORY CSV
# ================================================================================
def test_loss_history_csv_columns():
    """
    loss_history.csv must exist after training and contain the correct columns.
    This test checks the columns only — not that training has actually run.
    """
    csv_path = RESULTS_DIR / "loss_history.csv"
    if not csv_path.exists():
        pytest.skip("loss_history.csv not found — run training first")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        columns = reader.fieldnames

    required = {
        "epoch", "train_loss_total", "train_loss_recon",
        "train_loss_kl_g0", "train_loss_kl_u", "train_loss_cls",
        "val_loss_total", "val_loss_recon",
        "val_loss_kl_g0", "val_loss_kl_u", "val_loss_cls",
        "lr"
    }
    assert required == set(columns), \
        f"Missing columns: {required - set(columns)}"


def test_loss_history_csv_values_are_finite():
    """All loss values in loss_history.csv must be finite numbers."""
    csv_path = RESULTS_DIR / "loss_history.csv"
    if not csv_path.exists():
        pytest.skip("loss_history.csv not found — run training first")

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            for col in ["train_loss_total", "val_loss_total"]:
                val = float(row[col])
                assert np.isfinite(val), \
                    f"Non-finite value in {col} at epoch {row['epoch']}: {val}"


# ================================================================================
# 6. CHECKPOINT SAVE AND RELOAD
# ================================================================================
def test_best_model_checkpoint_exists():
    """best_model.pt must exist after training."""
    checkpoint_path = RESULTS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        pytest.skip("best_model.pt not found — run training first")
    assert checkpoint_path.exists()


def test_best_model_checkpoint_loadable():
    """best_model.pt must be loadable with torch.load."""
    checkpoint_path = RESULTS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        pytest.skip("best_model.pt not found — run training first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    required_keys = {"epoch", "model_state_dict", "optimizer_state_dict",
                     "val_loss_total", "train_loss_total"}
    assert required_keys == set(checkpoint.keys()), \
        f"Missing checkpoint keys: {required_keys - set(checkpoint.keys())}"


def test_checkpoint_val_loss_matches_csv():
    """
    The best validation loss stored in the checkpoint must match
    the minimum val_loss_total in loss_history.csv.
    """
    checkpoint_path = RESULTS_DIR / "best_model.pt"
    csv_path        = RESULTS_DIR / "loss_history.csv"

    if not checkpoint_path.exists() or not csv_path.exists():
        pytest.skip("Training outputs not found — run training first")

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    checkpoint_best = checkpoint["val_loss_total"]

    with open(csv_path) as f:
        reader = csv.DictReader(f)
        csv_best = min(float(row["val_loss_total"]) for row in reader)

    assert abs(checkpoint_best - csv_best) < 1e-4, \
        f"Checkpoint val loss {checkpoint_best:.6f} != CSV best {csv_best:.6f}"


def test_checkpoint_model_can_run_forward():
    """Reloaded checkpoint must produce valid forward pass output."""
    checkpoint_path = RESULTS_DIR / "best_model.pt"
    if not checkpoint_path.exists():
        pytest.skip("best_model.pt not found — run training first")

    from models.brick import BRICK
    model = BRICK()
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    with torch.no_grad():
        x = torch.randn(T_DATA, N_ROIS)
        out = model(x)

    assert out["x_recon"].shape == (T_DATA, N_ROIS)
    assert torch.isfinite(out["x_recon"]).all()