"""
================================================================================
LOSO (Leave-One-Subject-Out) Cross-Validation Study
================================================================================

Description:
    Runs K-fold leave-one-subject-out cross-validation across all 19 subjects
    in the LIFUS-BRICK dataset. For each fold:
        - One subject is held out entirely as the test set (all sessions /
          targets for that subject)
        - N_VAL_SUBJECTS other subjects are drawn (reproducibly, via
          split_seed) from the remaining pool to form the validation set
        - The rest form the training set

    This bypasses train.py's seed-based split_dataset() in favor of an
    explicit, subject-list-driven split (train.py's new `subject_split`
    param), while calling train.train() directly for the actual training
    loop -- so fold-to-fold training behavior is guaranteed identical to a
    normal train.py run, just with a different split.

    Outputs saved to results/loso_19_fold/fold_{subject_id}/, each folder
    containing the same artifacts a train.py run produces: best_model_recon.pt,
    best_model_cls.pt, final_model.pt, loss_history.csv, split.json.
    A top-level results/loso_19_fold/loso_summary.json records the fold
    definitions and best val loss per fold.

Usage:
    python training/loso_study.py
    python training/loso_study.py --n-val-subjects 3
    python training/loso_study.py --epochs 100                    # pilot run
    python training/loso_study.py --subjects sub-01 sub-02        # only these folds
    python training/loso_study.py --no-ic                         # ablation, all folds
"""

import sys
import json
import random
import argparse
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from training.train import train, DATA_DIR, N_EPOCHS, SEED
from training.dataset import BRICKDataset

LOSO_RESULTS_DIR      = ROOT_DIR / "results" / "training" / "loso_19_fold_beta_0.2"
N_VAL_SUBJECTS_DEFAULT = 5


# ================================================================================
# SUBJECT / SPLIT HELPERS
# ================================================================================
def get_all_subject_ids(ds) -> list:
    """Return sorted unique subject IDs present in the dataset."""
    return sorted({ds[i]["subject_id"] for i in range(len(ds))})


def make_loso_split(all_subjects: list, test_subject: str,
                     n_val_subjects: int, split_seed: int) -> dict:
    """
    Build a subject-level train/val/test split for one LOSO fold.

    test:  [test_subject] only
    val:   n_val_subjects subjects drawn reproducibly (random.Random(split_seed)
           shuffle) from the remaining pool, excluding test_subject
    train: everything else

    Note: split_seed is fixed across folds, but combined with the varying
    test_subject the resulting shuffle -- and therefore the val set -- differs
    fold to fold while staying reproducible on rerun.
    """
    remaining = [s for s in all_subjects if s != test_subject]

    if n_val_subjects >= len(remaining):
        raise ValueError(
            f"n_val_subjects={n_val_subjects} leaves no training subjects "
            f"out of {len(remaining)} remaining after holding out {test_subject}."
        )

    rng = random.Random(f"{split_seed}_{test_subject}")
    shuffled = remaining.copy()
    rng.shuffle(shuffled)

    val_subjects   = sorted(shuffled[:n_val_subjects])
    train_subjects = sorted(shuffled[n_val_subjects:])

    return {"train": train_subjects, "val": val_subjects, "test": [test_subject]}


# ================================================================================
# LOSO STUDY
# ================================================================================
def run_loso_study(
    n_val_subjects: int   = N_VAL_SUBJECTS_DEFAULT,
    n_epochs:       int   = N_EPOCHS,
    subjects:       list  = None,
    split_seed:     int   = SEED,
    train_seed:     int   = SEED,
    use_control:    bool  = True,
    use_ic:         bool  = True,
    lambda_noise:   float = None,
    weight_decay:   float = None,
    batch_size:     int   = None,
    beta:           float = None,
    epsilon:        float = None,
) -> dict:
    """
    Run one LOSO fold per subject (or per subject in `subjects`, if given),
    reusing training/train.py's train() for each fold.

    Returns:
        dict mapping test_subject -> best_val_loss for that fold
    """
    ds = BRICKDataset(DATA_DIR)
    all_subjects = get_all_subject_ids(ds)
    fold_subjects = subjects if subjects is not None else all_subjects

    unknown = set(fold_subjects) - set(all_subjects)
    if unknown:
        raise ValueError(f"Unknown subject IDs requested: {sorted(unknown)}")

    print(f"LOSO study: {len(fold_subjects)} fold(s) of {len(all_subjects)} total subjects | "
          f"N_VAL_SUBJECTS={n_val_subjects} | split_seed={split_seed}")

    fold_splits  = {}
    fold_results = {}

    for test_subject in fold_subjects:
        run_name = f"loso_19_fold/fold_{test_subject}"
        split = make_loso_split(all_subjects, test_subject, n_val_subjects, split_seed)
        fold_splits[test_subject] = split

        print(f"\n{'=' * 60}\nFold: {test_subject}  "
              f"(train={len(split['train'])}, val={len(split['val'])}, test=1)\n{'=' * 60}")

        best_val_loss = train(
            n_epochs=n_epochs,
            use_control=use_control,
            use_ic=use_ic,
            run_name=run_name,
            lambda_noise=lambda_noise,
            weight_decay=weight_decay,
            batch_size=batch_size,
            beta=beta,
            epsilon=epsilon,
            train_seed=train_seed,
            subject_split=split,
            base_results_dir=ROOT_DIR / "results",
        )
        fold_results[test_subject] = best_val_loss

    LOSO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = LOSO_RESULTS_DIR / "loso_summary.json"
    with open(summary_path, "w") as f:
        json.dump({
            "n_val_subjects":     n_val_subjects,
            "split_seed":         split_seed,
            "train_seed":         train_seed,
            "use_control":        use_control,
            "use_ic":             use_ic,
            "fold_splits":        fold_splits,
            "fold_best_val_loss": fold_results,
        }, f, indent=2)
    print(f"\nLOSO study complete. Summary saved to {summary_path}")

    return fold_results


# ================================================================================
# ENTRY POINT
# ================================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LOSO cross-validation study for BRICK")
    parser.add_argument("--n-val-subjects", type=int, default=N_VAL_SUBJECTS_DEFAULT,
                        help="Number of subjects held out for validation per fold (default: 5)")
    parser.add_argument("--epochs", type=int, default=N_EPOCHS)
    parser.add_argument("--subjects", type=str, nargs="+", default=None,
                        help="Run only these subject IDs as test folds (default: all subjects)")
    parser.add_argument("--split-seed", type=int, default=SEED,
                        help="Seed controlling which subjects are drawn into val per fold")
    parser.add_argument("--train-seed", type=int, default=SEED,
                        help="Seed controlling model init / training stochasticity")
    parser.add_argument("--no-control", action="store_true",
                        help="Ablation: disable control module")
    parser.add_argument("--no-ic", action="store_true",
                        help="Ablation: disable initial condition module")
    args = parser.parse_args()

    run_loso_study(
        n_val_subjects=args.n_val_subjects,
        n_epochs=args.epochs,
        subjects=args.subjects,
        split_seed=args.split_seed,
        train_seed=args.train_seed,
        use_control=not args.no_control,
        use_ic=not args.no_ic,
    )