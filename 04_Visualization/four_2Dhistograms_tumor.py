import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree


NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"


def user_continue_ans(user_choice:str = 'YN'):
    user_ans = input('Would you like to continue for another patient? (Y/N)').upper()
    while user_ans not in user_choice:
        user_ans = input('Invalid input. Please insert again: ').upper()
    return user_ans

def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    # Normalize input list to lowercase for comparison
    valid_answers = {mri.lower(): mri for mri in INPUT_MRI}

    while user_ans not in valid_answers:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    return valid_answers[user_ans]  # Return the correctly formatted value (T1/T1c/T2/FLAIR)

def get_patients_number():
    while True:
        try:
            patient_number = int(input("Enter patient number: "))
            if 1 <= patient_number <= 540:
                return str(patient_number).zfill(4)  # Ensures 4-digit formatting
            else:
                print("Patients go from 1 to 540. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")

def get_scatterplot_native_biasfield_tumor(mri_fname:str, array_mri:np.array, array_biasfield:np.array, brain_seg_array:np.array, tumor_mask_array:np.array, display = False, save = False, ax = None):
    if ax is None:
        fig, ax = plt.subplots()

        # Mask for valid brain voxels
    mask = array_mri > 0

    # Separate tumor and non-tumor masks
    if tumor_mask_array is not None:
        tumor_mask = (tumor_mask_array > 0) & mask
        non_tumor_mask = (~tumor_mask) & mask
    else:
        tumor_mask = np.zeros_like(array_mri, dtype=bool)
        non_tumor_mask = mask

    # Non-tumor: regular hexbin plot (density colormap)
    x_vals_non_tumor = array_mri[non_tumor_mask]
    y_vals_non_tumor = array_biasfield[non_tumor_mask]
    hxb = ax.hexbin(x_vals_non_tumor, y_vals_non_tumor,
                    gridsize=100, bins='log', cmap='viridis')

    # Tumor: overlay scatterplot in red
    if np.any(tumor_mask):
        x_vals_tumor = array_mri[tumor_mask]
        y_vals_tumor = array_biasfield[tumor_mask]
        ax.scatter(x_vals_tumor, y_vals_tumor, color='red', s=1, alpha=0.6, label='Tumor Voxels')
    # Add colorbar for non-tumor hexbin
    cbar = plt.colorbar(hxb, ax=ax)
    cbar.set_label('Non-Tumor Density')

    # Labels, title, and style
    ax.set_xlabel('Native MRI intensities', color='black')
    ax.set_ylabel('Biasfield intensities', color='black')
    ax.set_title(f'Scatterplot of Intensities (Rescaled): {mri_fname}', color='black')
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')
    ax.set_facecolor('black')
    ax.tick_params(axis='both', colors='black')

    if np.any(tumor_mask):
        legend = ax.legend(loc='upper right', fontsize=8, facecolor='white', edgecolor='white')
        for text in legend.get_texts():
            text.set_color('black')

    return ax

def compute_all_scatterplots_mri_type(new_dir_path, patient_dir_name_nifti, mri_type:str, patient_number, display=False, save=False):
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

    bias_n4_brain_nameplot = f'Patient {patient_number}: N4 w Brain on Brain'
    bias_n4_healthy_nameplot = f'Patient {patient_number}: N4 w Healthy Brain on Healthy Brain'
    bias_n4_brain_healthy_nameplot = f'Patient {patient_number}: N4 w Brain on Healthy Brain'
    bias_n4_healthy_brain_nameplot = f'Patient {patient_number}: N4 w Healthy Brain on Brain'


    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    get_scatterplot_native_biasfield_tumor(bias_n4_brain_nameplot, native_image_array, bias_n4_brain_array, brain_seg_array, tumor_binary_array, display=display, save=save, ax=axs[0, 0])
    get_scatterplot_native_biasfield_tumor(bias_n4_healthy_nameplot, native_image_array, bias_n4_healthy_array, brain_seg_array, tumor_binary_array, display=display, save=save, ax=axs[0, 1])
    get_scatterplot_native_biasfield_tumor(bias_n4_brain_healthy_nameplot, native_image_array, bias_n4_brain_healthy_array, brain_seg_array, tumor_binary_array, display=display, save=save, ax=axs[1, 0])
    get_scatterplot_native_biasfield_tumor(bias_n4_healthy_brain_nameplot, native_image_array, bias_n4_healthy_brain_array, brain_seg_array, tumor_binary_array, display=display, save=save, ax=axs[1, 1])

def continue_main_for_new_patient(NEW_DIR, folder_name, mri_type, patient_number, display=True, save=False):
    user_ans = user_continue_ans(user_choice= 'YN')
    while user_ans == 'Y':
        mri_type = get_user_answer(INPUT_MRI)
        patient_number = get_patients_number()
        compute_all_scatterplots_mri_type(NEW_DIR, f'UCSF-PDGM-{patient_number}_nifti', mri_type, patient_number,
                                              display=True, save=False)
        plt.tight_layout()
        plt.show()
        user_ans = user_continue_ans(user_choice='YN')

    print('Goodbye!')

# MAIN
if __name__ == '__main__':
    mri_type = get_user_answer(INPUT_MRI)
    patient_number = get_patients_number()
    compute_all_scatterplots_mri_type(NEW_DIR, f'UCSF-PDGM-{patient_number}_nifti', mri_type, patient_number,
                                      display=True, save=False)
    plt.tight_layout()
    plt.show()
    continue_main_for_new_patient(NEW_DIR, f'UCSF-PDGM-{patient_number}_nifti', mri_type, patient_number,
                                  display=True, save=False)