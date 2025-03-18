import os

import numpy as np
import matplotlib.pyplot as plt

from save_files import NEW_DIR
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"


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
                return str(patient_number)  # Convert to string for folder matching
            else:
                print("Patients go from 1 to 540. Try again.")
        except ValueError:
            print("Invalid input. Please enter a number.")


def calculate_tumor_histogram(mri_n4_fname: str, array_mri_n4: np.array, array_tumor_binary: np.array, bins_number=655,
                              display=False, save=False, ax=None):
    array_tumor_n4 = array_mri_n4 * array_tumor_binary

    hist_mri_n4, bins_mri_n4 = np.histogram(array_mri_n4, bins=bins_number, range=(0, 65536))
    hist_tumor_n4, bins_tumor_n4 = np.histogram(array_tumor_n4, bins=bins_number, range=(0, 65536))

    if ax is None:  # If no axes provided, create a new plot
        fig, ax = plt.subplots()

    # Plot histograms on the provided axis
    ax.plot(bins_mri_n4[1:-1], hist_mri_n4[1:], color='pink', linewidth=1)
    ax.plot(bins_tumor_n4[1:-1], hist_tumor_n4[1:], color='yellow', linewidth=1)

    # Labels and title in white
    ax.set_xlabel('Voxel Intensity', color='black')
    ax.set_ylabel('Frequency', color='black')
    ax.set_title(f'Histogram of Voxel Intensities (Rescaled) \n {mri_n4_fname}', color='black')

    # Grid with dashed white lines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Set dark background for the plot
    ax.set_facecolor('black')
    ax.tick_params(axis='both', colors='black')  # White ticks

    return ax  # Return the axis for future customization


def calculate_all_histograms_mri_type(new_dir_path, patient_dir_name_nifti, mri_type:str, display=False, save=False):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return
    patient_dir_name = patient_dir_name_nifti.split("_")[0]

    tumor_binary_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_tumor_segmentation_array.npy')
    tumor_binary_array = np.load(tumor_binary_array_path).astype(np.float32)

    mri_n4_brain_name = f'{patient_dir_name}_{mri_type}_N4_brain_rescaled'
    mri_n4_healthy_name = f'{patient_dir_name}_{mri_type}_N4_healthy_mask_rescaled'
    mri_n4_brain_healthy_name = f'{patient_dir_name}_{mri_type}_N4_brain_healthy_mask_rescaled'
    mri_n4_healthy_brain_name = f'{patient_dir_name}_{mri_type}_N4_healthy_mask_brain_rescaled'

    # Paths of the arrays
    mri_n4_brain_path = os.path.join(array_dir_path, f'{mri_n4_brain_name}.npy')
    mri_n4_healthy_path = os.path.join(array_dir_path, f'{mri_n4_healthy_name}.npy')
    mri_n4_brain_healthy_path = os.path.join(array_dir_path, f'{mri_n4_brain_healthy_name}.npy')
    mri_n4_healthy_brain_path = os.path.join(array_dir_path, f'{mri_n4_healthy_brain_name}.npy')

    mri_n4_brain_array = np.load(mri_n4_brain_path).astype(np.float32)
    mri_n4_healthy_array = np.load(mri_n4_healthy_path).astype(np.float32)
    mri_n4_brain_healthy_array = np.load(mri_n4_brain_healthy_path).astype(np.float32)
    mri_n4_healthy_brain_array = np.load(mri_n4_healthy_brain_path).astype(np.float32)

    fig, axs = plt.subplots(2, 2, figsize=(12,10))
    calculate_tumor_histogram(mri_n4_brain_name,mri_n4_brain_array,tumor_binary_array, display=display, save=save, ax = axs[0,0])
    calculate_tumor_histogram(mri_n4_healthy_name,mri_n4_healthy_array,tumor_binary_array, display=display, save=save, ax = axs[0,1])
    calculate_tumor_histogram(mri_n4_brain_healthy_name,mri_n4_brain_healthy_array,tumor_binary_array, display=display, save=save, ax = axs[1,0])
    calculate_tumor_histogram(mri_n4_healthy_brain_name,mri_n4_healthy_brain_array,tumor_binary_array, display=display, save=save, ax = axs[1,1])

# MAIN
if __name__ == '__main__':
    mri_type = get_user_answer(INPUT_MRI)
    patient_number = get_patients_number()
    calculate_all_histograms_mri_type(NEW_DIR, f'UCSF-PDGM-0{patient_number}_nifti', mri_type, display=True, save=False)
    plt.show()


