import glob
import scipy.io
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
PREPROCESSED_DIR = ROOT_DIR / "data" / "preprocessed_data"

TARGET_ROIS = [
    'lh_Ca', 'lh_GPe', 'lh_GPi', 'lh_Pu', 'lh_STH', 'lh_cerebellum_dentate',
    'lh_cerebellum_motor', 'lh_paracentral_smooth3mm', 'lh_postcentral_smooth3mm',
    'lh_precentral_smooth3mm', 'lh_superiorfrontal_smooth3mm', 'lh_vim', 'lh_zi',
    'rh_Ca', 'rh_GPe', 'rh_GPi', 'rh_Pu', 'rh_STH', 'rh_cerebellum_dentate',
    'rh_cerebellum_motor', 'rh_paracentral_smooth3mm', 'rh_postcentral_smooth3mm',
    'rh_precentral_smooth3mm', 'rh_superiorfrontal_smooth3mm'
]

def get_all_files():
    """Return sorted list of all preprocessed .mat file paths."""
    return sorted(glob.glob(str(PREPROCESSED_DIR / "*.mat")))

def load_subject(filepath):
    """
    Load a single preprocessed .mat file.
    Returns dict with keys: mpre, mpost, roi_names, subject_id, target, group
    """
    path = Path(filepath)
    mat = scipy.io.loadmat(filepath)
    parts = path.stem.split('_')
    
    return {
        'mpre':       mat['mpre'],
        'mpost':      mat['mpost'],
        'roi_names':  TARGET_ROIS,
        'subject_id': parts[0].replace('sub-', ''),
        'target':     parts[2].replace('roi-', ''),
        'group': str(mat['group'].flat[0]).strip("[]'\""),
        'filepath':   str(path)
    }

def load_all():
    """Load all preprocessed files. Returns list of subject dicts."""
    return [load_subject(f) for f in get_all_files()]