import numpy as np
import matplotlib.pyplot as plt
import os

NEW_DIR = "/mnt/external/reorg_patients_UCSF"


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
def plot_ax_mri_type(hist_mri, bins_mri, hist_tumor, bins_tumor, name_plot, ax = None):
    if ax is None:  # If no axes provided, create a new plot
        fig, ax = plt.subplots()

    # Plot histograms on the provided axis
    ax.plot(bins_mri[1:-1], hist_mri[1:], color='pink', linewidth=1)
    ax.plot(bins_tumor[1:-1], hist_tumor[1:], color='yellow', linewidth=1)

    # Labels and title in white
    ax.set_xlabel('Voxel Intensity', color='black')
    ax.set_ylabel('Frequency', color='black')
    ax.set_title(name_plot, color='black')

    # Grid with dashed white lines
    ax.grid(True, linestyle='--', linewidth=0.5, color='gray')

    # Set dark background for the plot
    ax.set_facecolor('black')
    ax.tick_params(axis='both', colors='black')  # White ticks

    return ax
def calculate_patient_histograms_native_tumor(patient_number, patient_folder_name, array_t1:np.array, array_t1c:np.array, array_t2:np.array, array_flair:np.array, array_tumor_binary:np.array, bins_number=655):
    # Check and compute arrays for tumor
    if patient_number in patient_folder_name:
        mri_t1_name_plot = f"Patient {patient_number}: T1"
        mri_t1c_name_plot = f"Patient {patient_number}: Tc"
        mri_t2_name_plot = f"Patient {patient_number}: T2"
        mri_flair_name_plot = f"Patient {patient_number}: FLAIR"

        array_t1_tumor = array_t1 * array_tumor_binary
        array_t1c_tumor = array_t1c * array_tumor_binary
        array_t2_tumor = array_t2 * array_tumor_binary
        array_flair_tumor = array_flair * array_tumor_binary


        hist_mri_t1, bins_mri_t1 = np.histogram(array_t1, bins=bins_number, range=(0, 65536))
        hist_mri_t1c, bins_mri_t1c = np.histogram(array_t1c, bins=bins_number, range=(0, 65536))
        hist_mri_t2, bins_mri_t2 = np.histogram(array_t2, bins=bins_number, range=(0, 65536))
        hist_mri_flair, bins_mri_flair = np.histogram(array_flair, bins=bins_number, range=(0, 65536))

        hist_tumor_t1, bins_tumor_t1 = np.histogram(array_t1_tumor, bins=bins_number, range=(0, 65536))
        hist_tumor_t1c, bins_tumor_t1c = np.histogram(array_t1c_tumor, bins=bins_number, range=(0, 65536))
        hist_tumor_t2, bins_tumor_t2 = np.histogram(array_t2_tumor, bins=bins_number, range=(0, 65536))
        hist_tumor_flair, bins_tumor_flair = np.histogram(array_flair_tumor, bins=bins_number, range=(0, 65536))

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        plot_ax_mri_type(hist_mri_t1, bins_mri_t1, hist_tumor_t1, bins_tumor_t1, mri_t1_name_plot, ax = axs[0,0])
        plot_ax_mri_type(hist_mri_t1c, bins_mri_t1c, hist_tumor_t1c, bins_tumor_t1c, mri_t1c_name_plot, ax = axs[0,1])
        plot_ax_mri_type(hist_mri_t2, bins_mri_t2, hist_tumor_t2, bins_tumor_t2, mri_t2_name_plot, ax = axs[1,0])
        plot_ax_mri_type(hist_mri_flair, bins_mri_flair, hist_tumor_flair, bins_tumor_flair, mri_flair_name_plot, ax = axs[1,1])

        return axs

def load_array_from_dir_patient(new_dir_path, patient_dir_name_nifti, patient_number):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return
    patient_dir_name = patient_dir_name_nifti.split("_")[0]

    tumor_binary_array_path = os.path.join(array_dir_path, f'{patient_dir_name}_tumor_binary_array.npy')
    tumor_binary_array = np.load(tumor_binary_array_path).astype(np.float32)

    mri_t1_name = f"{patient_dir_name}_T1_rescaled"
    mri_t1c_name = f"{patient_dir_name}_T1c_rescaled"
    mri_t2_name = f"{patient_dir_name}_T2_rescaled"
    mri_flair_name = f"{patient_dir_name}_FLAIR_rescaled"

    # Paths of the arrays
    mri_t1_path = os.path.join(array_dir_path, f'{mri_t1_name}.npy')
    mri_t1c_path = os.path.join(array_dir_path, f'{mri_t1c_name}.npy')
    mri_t2_path = os.path.join(array_dir_path, f'{mri_t2_name}.npy')
    mri_flair_path = os.path.join(array_dir_path, f'{mri_flair_name}.npy')

    mri_t1_array = np.load(mri_t1_path).astype(np.float32)
    mri_t1c_array = np.load(mri_t1c_path).astype(np.float32)
    mri_t2_array = np.load(mri_t2_path).astype(np.float32)
    mri_flair_array = np.load(mri_flair_path).astype(np.float32)

    return mri_t1_array, mri_t1c_array, mri_t2_array, mri_flair_array, tumor_binary_array



# MAIN
# MAIN
patient_number = get_patients_number()
patient_folder_name = f'UCSF-PDGM-{patient_number}_nifti'
arrays = load_array_from_dir_patient(NEW_DIR, patient_folder_name, patient_number)
if arrays is None:
    print("Patient data not found or missing arrays. Exiting.")
    exit()

mri_t1_array, mri_t1c_array, mri_t2_array, mri_flair_array, tumor_binary_array = arrays

# Plot the histograms
axs = calculate_patient_histograms_native_tumor(
    patient_number,
    patient_folder_name,
    mri_t1_array,
    mri_t1c_array,
    mri_t2_array,
    mri_flair_array,
    tumor_binary_array
)

# Show the plot
plt.tight_layout()
plt.show()
