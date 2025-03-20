import multiprocessing

import nibabel as nib
import numpy as np

import os
from save_files import CONTROL1, NEW_DIR


def rescale_to_16bit(array, epsilon=1e-6):
    """Rescales a NumPy array to the range [0, 65535] (16-bit), handling constant arrays safely."""
    old_min, old_max = np.min(array), np.max(array)

    if old_max - old_min < epsilon:  # Check if the range is too small (constant image)
        return np.full_like(array, 32767, dtype=np.uint16)  # Set to mid-range instead of zero

    return (65535 * (array - old_min) / (old_max - old_min)).astype(np.uint16)

def get_arrays_for_patient(folder_name, subfolder_dir, array_dir, nii_file):
    if nii_file.startswith(f'biasfield_{folder_name}'):  # original file corrected to not exclude the biasfields
        if nii_file.endswith("_dn.nii.gz"):  # Exclude _dn.nii.gz files
            print(f"Skipping file: {nii_file} (ends with _dn.nii.gz)")
            return  # Skip processing
        nii_file_name = nii_file.split('.')[0]
        print(f"Processing file: {nii_file}")

    # Arrays
        img = nib.load(os.path.join(subfolder_dir, nii_file))
        img_array = img.get_fdata()
        rescaled_array = rescale_to_16bit(img_array) # Convert to 16-bit integer
        rescaled_array_path = os.path.join(array_dir, f"{nii_file_name}_rescaled.npy")
        np.save(rescaled_array_path, rescaled_array)
    print(f"Arrays saved successfully for {subfolder_dir} to {array_dir}")


def process_folder(folder):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        # anat_dir = os.path.join(folder_path, "anat")
        reg_dir = os.path.join(folder_path, "reg")
        array_dir = os.path.join(folder_path, "array")
        # Ensure directories exist
        #os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(reg_dir, exist_ok=True)
        os.makedirs(array_dir, exist_ok=True)

        print("Processing reg directory")
        for nii_file in os.listdir(reg_dir):
            get_arrays_for_patient(folder_name, reg_dir, array_dir, nii_file)
        """print("Processing anat directory")
        for nii_file in os.listdir(anat_dir):
            get_arrays_for_patient(folder_name, anat_dir, array_dir, nii_file)"""

# MAIN PIPELINE
folders = os.listdir(NEW_DIR)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_folder, folders)
