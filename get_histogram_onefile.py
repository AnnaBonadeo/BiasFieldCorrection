import os
import numpy as np
import matplotlib.pyplot as plt
import subprocess

from save_files import NEW_DIR
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image to analyze: ").upper()
    while user_ans not in INPUT_MRI:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").upper()
    return user_ans

# Load file
user_ans_MRI = get_user_answer(INPUT_MRI)
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
            if user_ans_MRI in nii_file:
                nii_path = os.path.join(array_dir, nii_file)
                nii_file_array = np.load(nii_path).astype(np.float32)  # or np.float64

                # Compute histogram
                hist, bins = np.histogram(nii_file_array, bins=65536, range=(0, 65536))

                # Set up figure with black background
                plt.figure(figsize=(10, 6), facecolor='black')

                # Plot histogram with white lines
                plt.plot(bins[1:-1], hist[1:], color='white', linewidth=1)

                # Labels and title in white
                plt.xlabel('Voxel Intensity', color='white')
                plt.ylabel('Frequency', color='white')
                plt.title('Histogram of Voxel Intensities (Rescaled)', color='white')

                # Grid with dashed white lines
                plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

                # Set dark background for the plot
                plt.gca().set_facecolor('black')
                plt.tick_params(axis='both', colors='white')  # White ticks

                # Show plot
                plt.show()
                
                #fsleyes_command = f"fsleyes {nii_path}"
                #subprocess.run(fsleyes_command, shell=True)

        break

"""file_path = input("Enter file path: ")
file_array = np.load(file_path).astype(np.float32)  # or np.float64"""

"""# Compute histogram
hist, bins = np.histogram(file_array, bins=65536, range=(0, 65536))

# Set up figure with black background
plt.figure(figsize=(10, 6), facecolor='black')

# Plot histogram with white lines
plt.plot(bins[1:-1], hist[1:], color='white', linewidth=1)

# Labels and title in white
plt.xlabel('Voxel Intensity', color='white')
plt.ylabel('Frequency', color='white')
plt.title('Histogram of Voxel Intensities (Rescaled)', color='white')

# Grid with dashed white lines
plt.grid(True, linestyle='--', linewidth=0.5, color='gray')

# Set dark background for the plot
plt.gca().set_facecolor('black')
plt.tick_params(axis='both', colors='white')  # White ticks

# Show plot
plt.show()"""
