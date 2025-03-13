import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import subprocess

from save_files import NEW_DIR
CONTROL_TUMOR = "tumor_segmentation"
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

        
# Load file
user_ans_MRI = get_user_answer(INPUT_MRI)
print(f"Selected MRI Type: {user_ans_MRI}")
patient_number = get_patients_number()
print(f"Selected patient number: {patient_number}")

for folder in os.listdir(NEW_DIR):
    if patient_number in folder:
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(NEW_DIR, folder)
        array_dir = os.path.join(folder_path, "array")
        # Ensure directories exist
        os.makedirs(array_dir, exist_ok=True)
        # idea provare a trovare hist_tumor seg fuori dal for e poi fare lista degli altri histogrammi e salvare i nomi !!!!
        tumor_seg_data = []
        mri_data = {}
        for nii_file in os.listdir(array_dir):
            #if (CONTROL_TUMOR not in nii_file) or (user_ans_MRI not in nii_file):
                #continue
            if CONTROL_TUMOR in nii_file:
                tumor_seg_path = os.path.join(array_dir, nii_file)
                tumor_seg_array = np.load(tumor_seg_path).astype(np.float32)
                hist_tumor_seg, bins_tumor_seg = np.histogram(tumor_seg_array, bins=655, range=(0, 65536))

                # Fill in the list with data from tumor segmentation
                tumor_seg_data.append(hist_tumor_seg)
                print("Histogram added to tumor data")
                tumor_seg_data.append(bins_tumor_seg)
                print("Bins added to tumor data")

            if user_ans_MRI in nii_file:
                nii_file_name = nii_file.split('.')[0] # you need to remove rescaled from the name as well
                nii_path = os.path.join(array_dir, nii_file)
                nii_array = np.load(nii_path).astype(np.float32)
                hist_nii, bins_nii = np.histogram(nii_array, bins=655, range=(0, 65536))
                mri_data[nii_file_name] = hist_nii
        print(tumor_seg_data)
        print(mri_data)
        # Use Seaborn's dark style
        sns.set_style("dark")
        plt.figure(figsize=(10, 6), facecolor='black')

        # Plot histograms using Seaborn - Tumor
        hist_tumor_seg = tumor_seg_data[0]
        bins_nii = tumor_seg_data[1]
        #sns.lineplot(x=bins_nii[1:-1], y=hist_nii[1:], color='white', linewidth=1, label='Brain')
        sns.lineplot(x=bins_nii[:-1], y=hist_tumor_seg[1:], color='red', linewidth=1, label='Tumor')

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


