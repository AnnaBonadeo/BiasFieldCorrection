import os
import numpy as np
import matplotlib.pyplot as plt

CONTROLTSEG = "tumor_segmentation"
NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

def user_continue_ans(user_choice:str):
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

def get_scatterplot_native_biasfield(mri_fname:str, array_tumorsegmentation:np.array, array_biasfield:np.array, display = False, save = False, ax = None):
    # Try remove 0
    mask = array_tumorsegmentation > 0  # Only include voxels with non-zero native intensities
    x_vals = array_tumorsegmentation[mask]
    y_vals = array_biasfield[mask]
    """mask = brain_seg_array  # Only include voxels with non-zero native intensities
    x_vals = np.multiply(array_mri, mask)
    y_vals = array_biasfield"""
    hxb = ax.hexbin(x_vals, y_vals, C=None, gridsize=100, bins='log')

    if ax is None:
        fig, ax = plt.subplots()
    # Add colorbar to indicate density
    cbar = plt.colorbar(hxb, ax=ax)
    cbar.set_label('Density')

    # Labels and title in white
    ax.set_xlabel('Native MRI intensities', color='black')
    ax.set_ylabel('Biasfield intensities', color='black')
    ax.set_title(f'Scatterplot of Intensities (Rescaled): {mri_fname}', color='black')

    # Grid with dashed white lines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Set dark background for the plot
    ax.set_facecolor('black')
    ax.tick_params(axis='both', colors='black')  # White ticks

    return ax  # Return the axis for future customization

def compute_all_scatterplots_mri_type(new_dir_path, patient_dir_name_nifti, mri_type:str, patient_number, display=False, save=False):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return
    patient_dir_name = patient_dir_name_nifti.split("_")[0]

    # Tumor segmentation
    tumor_seg_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_{CONTROLTSEG}_not_rescaled.npy')
    tumor_seg_array = np.load(tumor_seg_array_path).astype(np.float32)

     # Biasfield images
    bias_n4_brain_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_not_rescaled'
    bias_n4_healthy_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_not_rescaled'
    bias_n4_brain_healthy_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_healthy_mask_not_rescaled'
    bias_n4_healthy_brain_name = f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_brain_not_rescaled'

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
    get_scatterplot_native_biasfield(bias_n4_brain_nameplot, tumor_seg_array, bias_n4_brain_array,display=display, save=save,ax=axs[0, 0])
    get_scatterplot_native_biasfield(bias_n4_healthy_nameplot, tumor_seg_array, bias_n4_healthy_array, display=display,save=save, ax=axs[0, 1])
    get_scatterplot_native_biasfield(bias_n4_brain_healthy_nameplot, tumor_seg_array, bias_n4_brain_healthy_array, display=display, save=save, ax=axs[1, 0])
    get_scatterplot_native_biasfield(bias_n4_healthy_brain_nameplot, tumor_seg_array, bias_n4_healthy_brain_array, display=display, save=save, ax=axs[1, 1])

def continue_main_for_new_patient(NEW_DIR, folder_name, mri_type, patient_number, display=True, save=False):
    user_ans = user_continue_ans('YN')
    while user_ans == 'Y':
        mri_type = get_user_answer(INPUT_MRI)
        patient_number = get_patients_number()
        compute_all_scatterplots_mri_type(NEW_DIR, f'UCSF-PDGM-{patient_number}_nifti', mri_type, patient_number,
                                              display=True, save=False)
        plt.tight_layout()
        plt.show()
        user_ans = user_continue_ans('YN')

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