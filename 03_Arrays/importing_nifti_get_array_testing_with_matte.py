import nibabel as nib
import numpy as np

import os


def rescale_to_16bit(array, epsilon=1e-6):
    """Rescales a NumPy array to the range [0, 65535] (16-bit), handling constant arrays safely."""
    old_min, old_max = np.min(array), np.max(array)

    if old_max - old_min < epsilon:  # Check if the range is too small (constant image)
        return np.full_like(array, 32767, dtype=np.uint16)  # Set to mid-range instead of zero

    return (65535 * (array - old_min) / (old_max - old_min)).astype(np.uint16)

def get_arrays_for_patient(folder_name, subfolder_dir, array_dir, nii_file):
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

FOLDER_NAME_TEST = "bias_field_correction_samples"
REG_DIR_TEST = os.path.join(FOLDER_NAME_TEST, "reg")
ARRAY_DIR_TEST = os.path.join(FOLDER_NAME_TEST, "array")
NII_FILE_NAME = "UCSF-PDGM-0371_T2_N4_healthy_mask.nii.gz"
get_arrays_for_patient(FOLDER_NAME_TEST, REG_DIR_TEST, ARRAY_DIR_TEST, NII_FILE_NAME)
#print(ARRAY_DIR_TEST)