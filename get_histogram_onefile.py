import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from save_files import NEW_DIR
from importing_tumorseg_array import CONTROL_TUMOR
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"


def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    # Normalize input list to lowercase for comparison
    valid_answers = {mri.lower(): mri for mri in INPUT_MRI}

    while user_ans not in valid_answers:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    return valid_answers[user_ans]  # Return the correctly formatted value (T1/T1c/T2/FLAIR)


# Load file
user_ans_MRI = get_user_answer(INPUT_MRI)
print(f"Selected MRI Type: {user_ans_MRI}")
patient_number = int(input("Enter patient number: "))
while patient_number not in range(1,540):
	print("Patients go from 1 to 540")
	patient_number = int(input("Enter another number: "))
patient_number = str(patient_number)

for folder in os.listdir(NEW_DIR):
    if patient_number in folder:
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(NEW_DIR, folder)
        array_dir = os.path.join(folder_path, "array")
        # Ensure directories exist
        os.makedirs(array_dir, exist_ok=True)
        for nii_file in os.listdir(array_dir):
            if CONTROL_TUMOR in nii_file:
                tumor_seg_path = os.path.join(array_dir, nii_file)
                tumor_seg_array = np.load(tumor_seg_path).astype(np.float32)
            if user_ans_MRI in nii_file:
                nii_path = os.path.join(array_dir, nii_file)
                nii_file_array = np.load(nii_path).astype(np.float32)  # or np.float64

                # Compute histogram
                hist_nii, bins_nii = np.histogram(nii_file_array, bins=655, range=(0, 65536))
                hist_tumor_seg, _ = np.histogram(nii_file_array, bins=655, range=(0, 65536))

                # Use Seaborn's dark style
                sns.set_style("dark")
                plt.figure(figsize=(10, 6), facecolor='black')

                # Plot histograms using Seaborn
                sns.lineplot(x=bins_nii[1:-1], y=hist_nii[1:], color='white', linewidth=1, label='Brain')
                sns.lineplot(x=bins_nii[1:-1], y=hist_tumor_seg[1:], color='red', linewidth=1, label='Tumor')

                # Labels and title
                plt.xlabel('Voxel Intensity', color='white')
                plt.ylabel('Frequency', color='white')
                plt.title('Histogram of Voxel Intensities (Rescaled)', color='white')
                plt.legend(loc='upper right')

                # Set dark background
                plt.gca().set_facecolor('black')
                plt.tick_params(axis='both', colors='white')
                plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

                # Show plot
                plt.show()

        break


