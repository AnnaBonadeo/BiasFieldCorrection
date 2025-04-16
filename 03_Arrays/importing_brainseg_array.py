import nibabel as nib
import numpy as np
import multiprocessing
import os
from save_files import CONTROL1, NEW_DIR

CONTROL_BRAINSEG = "brain_segmentation"

def get_array_tumor_seg(folder_name, subfolder_dir, array_dir, nii_file):
    if nii_file.startswith(folder_name) and CONTROL_BRAINSEG in nii_file:
        nii_file_name = nii_file.split('.')[0]
        print(f"Processing file: {nii_file}")

        # Arrays
        img = nib.load(os.path.join(subfolder_dir, nii_file))
        img_array = img.get_fdata()
        img_array_path = os.path.join(array_dir, f"{nii_file_name}_array.npy")
        np.save(img_array_path, img_array)
        print(f"Tumor segmentation array successfully save for {subfolder_dir} to {array_dir}")
def process_folder(folder):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        seg_dir = os.path.join(folder_path, "seg")
        array_dir = os.path.join(folder_path, "array")
        # Ensure directories exist
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(array_dir, exist_ok=True)

        for nii_file in os.listdir(seg_dir):
            if nii_file.startswith(folder_name) and CONTROL_BRAINSEG in nii_file:
                get_array_tumor_seg(folder_name, seg_dir, array_dir, nii_file)
                break
# MAIN PIPELINE
folders = os.listdir(NEW_DIR)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_folder, folders)