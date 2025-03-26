import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde


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

def get_scatterplot_with_densities(x:np.array, y:np.array, bins = 655):
    x = x.ravel()
    y = y.ravel()
    # Calculate the point density
    # 1 # Compute the 2D histogram
    density, xedges, yedges = np.histogram2d(x, y, bins=bins)

    """# Convert bin edges into bin centers
    xcenters = (xedges[:-1] + xedges[1:]) / 2
    ycenters = (yedges[:-1] + yedges[1:]) / 2"""

    # Assign density values to each (x, y) point
    x_bin_indices = np.digitize(x, xedges) - 1
    y_bin_indices = np.digitize(y, yedges) - 1

    # Ensure indices stay within valid range
    x_bin_indices = np.clip(x_bin_indices, 0, bins - 1)
    y_bin_indices = np.clip(y_bin_indices, 0, bins - 1)

    # Assign density values to points
    z = density[x_bin_indices, y_bin_indices]
    return x, y, z


def calculate_scatterplot_biasfield_native(mri_fname, native_mri_array:np.array,biasfield_array:np.array,display = False, save = False, ax = None):
    # Checking sizes
    shape_native = np.shape(native_mri_array)
    shape_biasfield = np.shape(biasfield_array)
    print("Native ", shape_native, "Biasfield ", shape_biasfield)

    # SAMPLING for better visualization
    # Create a single mask (Boolean) that will be applied to both arrays
    sampling_rate = 0.1  # 10% sampling
    mask = np.random.rand(*native_mri_array.shape) < sampling_rate  # ✅ Same mask for both

    # Apply the same mask to both arrays
    sampled_native_mri_array = native_mri_array[mask]  # ✅ Corresponding values
    sampled_biasfield_array = biasfield_array[mask]  # ✅ Corresponding values

    if ax is None:
        fig, ax = plt.subplots()

    # If SAMPLING is removed
    #ax.scatter(native_mri_array,biasfield_array, s=1, alpha=0.5)
    # Plot the sampled points
    density_sampled_native_mri_array, density_sampled_biasfield_array, colors = get_scatterplot_with_densities(sampled_native_mri_array, sampled_biasfield_array)
    sc = ax.scatter(density_sampled_native_mri_array, density_sampled_biasfield_array, c = colors, cmap = 'viridis', s = 1, alpha = 0.5)

    # Add colorbar to indicate density
    cbar = plt.colorbar(sc, ax=ax)
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
    # Native image
    native_image_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_{mri_type}_rescaled.npy')
    native_image_array = np.load(native_image_array_path).astype(np.float32)

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
    calculate_scatterplot_biasfield_native(bias_n4_brain_nameplot, bias_n4_brain_array, native_image_array, display=display, save=save,ax=axs[0, 0])
    calculate_scatterplot_biasfield_native(bias_n4_healthy_nameplot, bias_n4_healthy_array, native_image_array, display=display,save=save, ax=axs[0, 1])
    calculate_scatterplot_biasfield_native(bias_n4_brain_healthy_nameplot, bias_n4_brain_healthy_array, native_image_array,display=display, save=save, ax=axs[1, 0])
    calculate_scatterplot_biasfield_native(bias_n4_healthy_brain_nameplot, bias_n4_healthy_brain_array, native_image_array, display=display, save=save, ax=axs[1, 1])

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