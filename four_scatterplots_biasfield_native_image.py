import numpy as np
import matplotlib.pyplot as plt

def calculate_scatterplot_biasfield_native(native_mri_array:np.array,biasfield_array:np.array, mri_fname:str, ax = None):
    if ax is None:
        fig, ax = plt.subplots()
    ax.scatter(native_mri_array,biasfield_array)

    # Labels and title in white
    ax.set_xlabel('Biasfield intensities', color='black')
    ax.set_ylabel('Native MRI intensities', color='black')
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

    biasfield_T1_array_path = os.path.join(array_dir_path, f'biasfield_{patient_dir_name}_rescaled.npy')