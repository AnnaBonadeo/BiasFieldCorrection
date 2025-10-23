import numpy as np
import os
import matplotlib.pyplot as plt

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
CONTROL1 = "UCSF-PDGM-"
MRI_TYPE = ["T1", "T1c", "T2", "FLAIR"]
BIASFIELD = ["N4BB", "N4HH", "N4BH", "N4HB"]



def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    # Normalize input list to lowercase for comparison
    valid_answers = {mri.lower(): mri for mri in INPUT_MRI}

    while user_ans not in valid_answers:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    return valid_answers[user_ans]  # Return the correctly formatted value (T1/T1c/T2/FLAIR)

# For each corrected image compute COM brain and COM tumor
def compute_center_of_mass_regions(array_mri: np.ndarray, array_biasfield: np.ndarray, brain_seg_array: np.ndarray, tumor_mask_array: np.ndarray, bins: int = 100):

    # Masks
    valid_mask = array_mri > 0
    tumor_mask = (tumor_mask_array > 0) & valid_mask
    brain_only_mask = (brain_seg_array > 0) & (~tumor_mask) & valid_mask

    def compute_center_of_mass(x, y, bins=100):
        if len(x) == 0 or len(y) == 0:
            return (np.nan, np.nan)
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters, indexing='ij')
        total = hist.sum()
        if total == 0:
            return (np.nan, np.nan)
        com_x = (X * hist).sum() / total
        com_y = (Y * hist).sum() / total
        return (com_x, com_y)

    # Extract values for each region
    x_brain = array_mri[brain_only_mask]
    y_brain = array_biasfield[brain_only_mask]
    x_tumor = array_mri[tumor_mask]
    y_tumor = array_biasfield[tumor_mask]

    # Compute CoM for brain-only and tumor-only
    com_brain = compute_center_of_mass(x_brain, y_brain, bins=bins)
    com_tumor = compute_center_of_mass(x_tumor, y_tumor, bins=bins)

    return com_brain, com_tumor

def compute_all_com_patient(new_dir_path, patient_dir_name_nifti, patient_number):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return
    patient_dir_name = patient_dir_name_nifti.split("_")[0]
    # Native image T1
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_T1_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

    # Native image T1c
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_T1c_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

    # Native image T2
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_T2_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

    # Native image FLAIR
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_FLAIR_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

    # Brain segmentation
    brain_seg_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_brain_segmentation_array.npy')
    brain_seg_array = np.load(brain_seg_array_path).astype(np.float32)

    # Tumor mask
    tumor_binary_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_tumor_binary_array.npy')
    tumor_binary_array = np.load(tumor_binary_array_path).astype(np.float32)

    # Native images names
    t1_image_name = f'Patient_{patient_number}_T1'
    t1c_image_name = f'Patient_{patient_number}_T1c'
    t2_image_name = f'Patient_{patient_number}_T2'
    flair_image_name = f'Patient_{patient_number}_FLAIR'



    # Paths of the arrays
    bias_n4_brain_path = os.path.join(array_dir_path, f'{bias_n4_brain_name}.npy')
    bias_n4_healthy_path = os.path.join(array_dir_path, f'{bias_n4_healthy_name}.npy')
    bias_n4_brain_healthy_path = os.path.join(array_dir_path, f'{bias_n4_brain_healthy_name}.npy')
    bias_n4_healthy_brain_path = os.path.join(array_dir_path, f'{bias_n4_healthy_brain_name}.npy')

    bias_n4_brain_array = np.load(bias_n4_brain_path).astype(np.float32)
    bias_n4_healthy_array = np.load(bias_n4_healthy_path).astype(np.float32)
    bias_n4_brain_healthy_array = np.load(bias_n4_brain_healthy_path).astype(np.float32)
    bias_n4_healthy_brain_array = np.load(bias_n4_healthy_brain_path).astype(np.float32)

    bias_n4_brain_nameplot = f'Patient_{patient_number}_N4BB'
    bias_n4_healthy_nameplot = f'Patient_{patient_number}_N4HH'
    bias_n4_brain_healthy_nameplot = f'Patient_{patient_number}_N4BH'
    bias_n4_healthy_brain_nameplot = f'Patient_{patient_number}_N4HB'

    com_brain_bb, com_tumor_bb = compute_center_of_mass_regions(native_image_array,
                                                                bias_n4_brain_array,
                                                                brain_seg_array,
                                                                tumor_binary_array)
    com_brain_hh, com_tumor_hh = compute_center_of_mass_regions(native_image_array,
                                                                bias_n4_healthy_array,
                                                                brain_seg_array,
                                                                tumor_binary_array)
    com_brain_bh, com_tumor_bh = compute_center_of_mass_regions(native_image_array,
                                                                bias_n4_brain_healthy_array,
                                                                brain_seg_array,
                                                                tumor_binary_array)
    com_brain_hb, com_tumor_hb = compute_center_of_mass_regions(native_image_array,
                                                                bias_n4_healthy_brain_array,
                                                                brain_seg_array,
                                                                tumor_binary_array)
    com_bb = {bias_n4_brain_nameplot: (com_brain_bb, com_tumor_bb),}
    com_hh = {bias_n4_healthy_nameplot: (com_brain_hh, com_tumor_hh)}
    com_bh = {bias_n4_brain_healthy_nameplot: (com_brain_bh, com_tumor_bh)}
    com_hb = {bias_n4_healthy_brain_nameplot: (com_brain_hb, com_tumor_hb)}
    all_com_mri_type_one_patient = {}
    all_com_mri_type_one_patient.update(com_bb)
    all_com_mri_type_one_patient.update(com_hh)
    all_com_mri_type_one_patient.update(com_bh)
    all_com_mri_type_one_patient.update(com_hb)
    return all_com_mri_type_one_patient