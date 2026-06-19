"""
================================================================================
BRICKDataset
================================================================================

Description:
    PyTorch Dataset for the LIFUS-BRICK project.

    Each .mat file in data/preprocessed_data/ contains one subject's BOLD
    timeseries for one sonication target (VIM or ZI). Each file contributes
    two items to the dataset — one for the pre-sonication session (mpre) and
    one for the post-sonication session (mpost).

    Each item is a dict:
        x               (torch.Tensor): BOLD timeseries, shape (T, N), float32
        fc              (torch.Tensor): Pearson FC matrix, shape (N, N), float32
        lifus_condition (torch.Tensor): 0=pre, 1=post (used by BRICK classifier)
        first_target    (torch.Tensor): 0=VIM_first, 1=ZI_first (analysis only)
        subject_id      (str):          e.g. 'sub-fuspd01'
        target          (str):          'vim' or 'zi'
        condition_str   (str):          'mpre' or 'mpost'
        group_str       (str):          'VIM_first' or 'ZI_first'

    Total items: n_files × 2 = 38 × 2 = 76

Usage:
    from training.dataset import BRICKDataset, split_dataset

    ds = BRICKDataset(data_dir)
    train_ds, val_ds, test_ds = split_dataset(ds)
"""

import random
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import Dataset, Subset
from scipy.io import loadmat

from models.koopman_utils import compute_pearson


class BRICKDataset(Dataset):
    """
    PyTorch Dataset wrapping preprocessed .mat files for LIFUS-BRICK.

    Args:
        data_dir (Path | str): Path to data/preprocessed_data/
        cache_fc (bool):       If True, precompute and cache all FC matrices
                               on first access. Saves compute during training
                               at the cost of memory. Default True.
    """

    def __init__(self, data_dir: Path, cache_fc: bool = True):
        self.data_dir = Path(data_dir)
        self.cache_fc = cache_fc

        # Build index: one entry per (file, condition) pair
        self._index = []           # list of (filepath, condition_str)
        self._fc_cache = {}        # filepath -> fc tensor (if cache_fc)

        mat_files = sorted(self.data_dir.glob("*.mat"))
        if not mat_files:
            raise FileNotFoundError(
                f"No .mat files found in {self.data_dir}. "
                "Run preprocessing/combine_and_align_rois.py first."
            )

        for filepath in mat_files:
            self._index.append((filepath, "mpre"))
            self._index.append((filepath, "mpost"))

    def __len__(self) -> int:
        return len(self._index)

    def __getitem__(self, idx: int) -> dict:
        filepath, condition_str = self._index[idx]

        # --- Load .mat file ---
        mat = loadmat(str(filepath))
        x_np = mat[condition_str]                       # (T, N) numpy float64

        # --- Convert to tensor ---
        x = torch.tensor(x_np, dtype=torch.float32)    # (T, N)

        # --- Z-score normalize each ROI (column) ---
        x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-8)

        # --- Compute or retrieve FC ---
        if self.cache_fc:
            if filepath not in self._fc_cache:
                self._fc_cache[filepath] = {}
            if condition_str not in self._fc_cache[filepath]:
                self._fc_cache[filepath][condition_str] = compute_pearson(x)
            fc = self._fc_cache[filepath][condition_str]
        else:
            fc = compute_pearson(x)                     # (N, N)

        # --- Labels ---
        lifus_condition = torch.tensor(
            0 if condition_str == "mpre" else 1,
            dtype=torch.long,
        )

        # Parse group from .mat file
        group_raw = mat["group"].flat[0]
        group_str = str(group_raw).strip().strip("[]'\"")
        first_target = torch.tensor(
            0 if group_str == "VIM_first" else 1,
            dtype=torch.long,
        )

        # --- Metadata ---
        filename   = filepath.stem                      # e.g. sub-fuspd01_ses-2_roi-vim_meants
        subject_id = filename.split("_")[0]             # e.g. sub-fuspd01
        target     = "vim" if "roi-vim" in filename.lower() else "zi"

        return {
            "x":               x,
            "fc":              fc,
            "lifus_condition": lifus_condition,
            "first_target":    first_target,
            "subject_id":      subject_id,
            "target":          target,
            "condition_str":   condition_str,
            "group_str":       group_str,
        }


def split_dataset(
    dataset:     BRICKDataset,
    train_ratio: float = 0.7,
    val_ratio:   float = 0.1,
    seed:        int   = 42,
) -> Tuple[Subset, Subset, Subset]:
    """
    Split BRICKDataset into train, val, and test subsets by subject ID.

    All sessions for a given subject (VIM pre, VIM post, ZI pre, ZI post)
    are kept together in the same split to prevent data leakage.

    Args:
        dataset     (BRICKDataset): The full dataset
        train_ratio (float):        Proportion for training. Default 0.7
        val_ratio   (float):        Proportion for validation. Default 0.1
        seed        (int):          Random seed for reproducibility. Default 42

    Returns:
        train_subset, val_subset, test_subset (torch.utils.data.Subset)
    """
    # Get unique subject IDs
    unique_subjects = sorted(set(
        dataset[i]["subject_id"] for i in range(len(dataset))
    ))

    # Shuffle subjects
    rng = random.Random(seed)
    rng.shuffle(unique_subjects)

    # Split subject IDs
    n          = len(unique_subjects)
    n_train    = int(n * train_ratio)
    n_val      = int(n * val_ratio)

    train_subjects = set(unique_subjects[:n_train])
    val_subjects   = set(unique_subjects[n_train:n_train + n_val])
    test_subjects  = set(unique_subjects[n_train + n_val:])

    # Map subject IDs to dataset indices
    train_indices, val_indices, test_indices = [], [], []
    for i in range(len(dataset)):
        sid = dataset[i]["subject_id"]
        if sid in train_subjects:
            train_indices.append(i)
        elif sid in val_subjects:
            val_indices.append(i)
        else:
            test_indices.append(i)

    return (
        Subset(dataset, train_indices),
        Subset(dataset, val_indices),
        Subset(dataset, test_indices),
    )