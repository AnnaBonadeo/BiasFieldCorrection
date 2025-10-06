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

# Iterate for all corrections for each patient
def compute_all_com_mri_type(new_dir_path, patient_dir_name_nifti, mri_type:str, patient_number):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return
    patient_dir_name = patient_dir_name_nifti.split("_")[0]
    # Native image
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_{mri_type}_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

    # Brain segmentation
    brain_seg_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_brain_segmentation_array.npy')
    brain_seg_array = np.load(brain_seg_array_path).astype(np.float32)

    # Tumor mask
    tumor_binary_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_tumor_binary_array.npy')
    tumor_binary_array = np.load(tumor_binary_array_path).astype(np.float32)

    # Biasfield images
    bias_n4_brain_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_rescaled'
    bias_n4_healthy_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_rescaled'
    bias_n4_brain_healthy_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_healthy_mask_rescaled'
    bias_n4_healthy_brain_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_brain_rescaled'

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

def compute_coms_all_patients(NEW_DIR, mri_type):
    # iterate for each patient
    all_com_bb = {}
    all_com_hh = {}
    all_com_bh = {}
    all_com_hb = {}
    for folder in os.listdir(NEW_DIR):
        if not (os.path.isdir(os.path.join(NEW_DIR, folder)) and folder.startswith(CONTROL1)):
            continue

        patient_number = folder.split("-")[2].split("_")[0]
        print(f"Processing patient {patient_number}")

        array_dir = os.path.join(NEW_DIR, folder, "array")
        if not os.path.exists(array_dir):
            print(f"Array directory missing for patient {patient_number}")
            continue

        all_com_patient = compute_all_com_mri_type(NEW_DIR, folder, mri_type, patient_number)
        for correction in all_com_patient: # correction = key in the dict
            if 'N4BB' in correction:
                all_com_bb[f'{patient_number}'] = all_com_patient[correction]
            elif 'N4HH' in correction:
                all_com_hh[f'{patient_number}'] = all_com_patient[correction]
            elif 'N4BH' in correction:
                all_com_bh[f'{patient_number}'] = all_com_patient[correction]
            elif 'N4HB' in correction:
                all_com_hb[f'{patient_number}'] = all_com_patient[correction]
    return all_com_bb, all_com_hh, all_com_bh, all_com_hb

def plot_coms_for_all_patients(all_com_bb, all_com_hh, all_com_bh, all_com_hb):


    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()  # flatten to 1D array for easy indexing
    corrections = ['N4BB', 'N4HH', 'N4BH', 'N4HB']
    all_com_dicts = [all_com_bb, all_com_hh, all_com_bh, all_com_hb]
    colors = {'brain': 'blue', 'tumor': 'red'}

    for i, (correction, com_dict) in enumerate(zip(corrections, all_com_dicts)):
        ax = axes[i]
        brain_points = [v[0] for v in com_dict.values()]
        tumor_points = [v[1] for v in com_dict.values()]

        # Convert to arrays for plotting
        brain_points = np.array(brain_points)
        tumor_points = np.array(tumor_points)

        if len(brain_points) == 0 or len(tumor_points) == 0:
            ax.set_title(f'CoM Scatterplot: {correction} (no valid data)')
            continue

        ax.scatter(brain_points[:, 0], brain_points[:, 1], color=colors['brain'], label='Brain-only', alpha=0.6)
        ax.scatter(tumor_points[:, 0], tumor_points[:, 1], color=colors['tumor'], label='Tumor-only', alpha=0.6)

        ax.set_title(f'CoM Scatterplot: {correction}')
        ax.set_xlabel('Native MRI intensity')
        ax.set_ylabel('Biasfield intensity')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()

# MAIN
"""if __name__ == '__main__':
    mri_type = get_user_answer(MRI_TYPE)
    all_com_bb, all_com_hh, all_com_bh, all_com_hb = compute_coms_all_patients(NEW_DIR, mri_type)
    plot_coms_for_all_patients(all_com_bb, all_com_hh, all_com_bh, all_com_hb)"""

# MAIN
if __name__ == '__main__':
    import concurrent.futures

    mri_type = get_user_answer(MRI_TYPE)

    # --- Collect all patient folders ---
    patient_folders = [
        folder for folder in os.listdir(NEW_DIR)
        if os.path.isdir(os.path.join(NEW_DIR, folder)) and folder.startswith(CONTROL1)
    ]

    # --- Define a helper to process each patient (wrapper) ---
    def process_patient(folder):
        patient_number = folder.split("-")[2].split("_")[0]
        array_dir = os.path.join(NEW_DIR, folder, "array")
        if not os.path.exists(array_dir):
            print(f"Array directory missing for patient {patient_number}")
            return patient_number, None
        try:
            result = compute_all_com_mri_type(NEW_DIR, folder, mri_type, patient_number)
            return patient_number, result
        except Exception as e:
            print(f"Error processing patient {patient_number}: {e}")
            return patient_number, None

    # --- Run in parallel ---
    all_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_patient, folder) for folder in patient_folders]
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())

    # --- Merge results ---
    all_com_bb, all_com_hh, all_com_bh, all_com_hb = {}, {}, {}, {}
    for patient_number, all_com_patient in all_results:
        if all_com_patient is None:
            continue
        for correction in all_com_patient:
            if 'N4BB' in correction:
                all_com_bb[patient_number] = all_com_patient[correction]
            elif 'N4HH' in correction:
                all_com_hh[patient_number] = all_com_patient[correction]
            elif 'N4BH' in correction:
                all_com_bh[patient_number] = all_com_patient[correction]
            elif 'N4HB' in correction:
                all_com_hb[patient_number] = all_com_patient[correction]

    # --- Plot ---
    plot_coms_for_all_patients(all_com_bb, all_com_hh, all_com_bh, all_com_hb)

