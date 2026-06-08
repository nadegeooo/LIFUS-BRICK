"""
================================================================================
Neuroimaging BOLD Signal Preprocessing Script: ROI Combination and Alignment
================================================================================

Description:
    This script streamlines the preprocessing of self-contained subject fMRI 
    BOLD time-series data stored in MATLAB (.mat) format. Each raw file contains 
    23 ROIs along with an isolated 24th sonication region stored under a separate key. 
    This function integrates the 25th region into the main data matrix and aligns
    features to a standardized 24-ROI configuration, preserving the original data layout.

Core Processing Steps:
    1. Scan the raw data directory for standard MATLAB (*.mat) files.
    2. Determine the label of the 24th region ('lh_zi' or 'lh_vim')
       by extracting substring signatures directly from the filename.
    3. Unnest MATLAB multi-dimensional character arrays from the 'fn' matrix into 
       Python strings to map existing column locations.
    4. Horizontally stack (np.hstack) the isolated 'target_meants_pre' and 
       'target_meants_post' arrays onto the main 23-column 'mpre' and 'mpost' matrices.
    5. Reorder the columns to guarantee an exact index match with the target 24-ROI template, 
       ensuring features mean the same thing across all subjects.
    6. Convert the standardized target string list into a NumPy object array to cleanly 
       emulate a MATLAB Cell Array structure upon export.
    7. Export the modified fields into a new .mat file under the target folder, 
       stripping out the redundant source keys to optimize file weight.

Matrix Constraints:
    Input Shape:  (T=240, N=23) for 'mpre'/'mpost', and (T=240, N=1) for targets.
    Output Shape: (T=240, N=24) for both 'mpre/mpost'.
    Orientation:  Maintains original structural orientation (Rows = Timepoints, Columns = ROIs).

Directory Structure Dependencies:
    - Source: data/raw_data/*.mat
    - Destination: data/preprocessed_data/*.mat

Execution:
    python preprocessing/combine_and_align_rois.py
"""

import os
import glob
import numpy as np
from scipy.io import loadmat, savemat

# --- PATHS ---
RAW_DIR = os.path.join("data", "raw_data")
OUTPUT_DIR = os.path.join("data", "preprocessed_data")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---  TARGET 24 ROIs IN ORDER ---
TARGET_ROIS = [
    'lh_Ca', 'lh_GPe', 'lh_GPi', 'lh_Pu', 'lh_STH', 'lh_cerebellum_dentate',
    'lh_cerebellum_motor', 'lh_paracentral_smooth3mm', 'lh_postcentral_smooth3mm',
    'lh_precentral_smooth3mm', 'lh_superiorfrontal_smooth3mm', 'lh_vim', 'lh_zi',
    'rh_Ca', 'rh_GPe', 'rh_GPi', 'rh_Pu', 'rh_STH', 'rh_cerebellum_dentate',
    'rh_cerebellum_motor', 'rh_paracentral_smooth3mm', 'rh_postcentral_smooth3mm',
    'rh_precentral_smooth3mm', 'rh_superiorfrontal_smooth3mm'
]

def preprocess_subject_data():
    # Find all main subject files
    mat_files = glob.glob(os.path.join(RAW_DIR, "*.mat"))
    
    if not mat_files:
        print(f"❌ No .mat files found in {RAW_DIR}!")
        return

    print(f"Found {len(mat_files)} files.\n")

    for filepath in mat_files:
        filename = os.path.basename(filepath)
        file_base = os.path.splitext(filename)[0]  # drop '.mat'
        
        print(f"Processing {filename}...")
        
        # 1. Determine the identity of the 24th column from the file name (vim or zi)
        if "roi-zi" in filename.lower():
            extra_label = "lh_zi"
        elif "roi-vim" in filename.lower():
            extra_label = "lh_vim"
        else:
            print(f"⚠️ Skipping {filename}: Name doesn't contain 'roi-zi' or 'roi-vim'.")
            continue
            
        # 2. Load the data fields
        mat_data = loadmat(filepath)
        mpre_23 = mat_data['mpre']
        mpost_23 = mat_data['mpost']
        target_pre = mat_data['target_meants_pre']
        target_post = mat_data['target_meants_post']
        
        # 3. Cleanly parse the existing 23 labels from the 'fn' matrix
        raw_labels = []
        for item in mat_data['fn'].flatten():
            while isinstance(item, np.ndarray):
                item = item[0] if item.size > 0 else ""
            raw_labels.append(str(item).strip())
            
        # 4. Append the target_meants column directly onto the end of the 23 columns
        mpre_24 = np.hstack((mpre_23, target_pre))
        mpost_24 = np.hstack((mpost_23, target_post))
        all_current_labels = raw_labels + [extra_label]
        
        # 5. Reorder columns to match your exact TARGET_ROIS order
        try:
            reorder_indices = [all_current_labels.index(roi) for roi in TARGET_ROIS]
        except ValueError as e:
            print(f"❌ Label matching error in {filename}: {e}")
            print(f"Available labels inside file were: {all_current_labels}")
            continue
            
        mpre_aligned = mpre_24[:, reorder_indices]
        mpost_aligned = mpost_24[:, reorder_indices]
        
        # 6. Convert TARGET_ROIS list to a MATLAB-compatible object array for 'fn'
        fn_aligned = np.array(TARGET_ROIS, dtype=object)
        
        # 7. Construct the new .mat dictionary containing ONLY the cleaned keys
        # Format is identical to input: shapes remain (240, 24)
        output_mat_data = {
            'mpre': mpre_aligned,
            'mpost': mpost_aligned,
            'fn': fn_aligned
        }
        
        # Save out as a standard .mat file
        output_filepath = os.path.join(OUTPUT_DIR, filename)
        savemat(output_filepath, output_mat_data)
        
        print(f"   Success! Saved {filename} with mpre/mpost shape: {mpre_aligned.shape}")
        
    print("\nProcessing complete! Cleaned .mat files are stored in data/preprocessed_data/")

if __name__ == "__main__":
    preprocess_subject_data()