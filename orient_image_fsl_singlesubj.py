import os
import subprocess

# Define the folder path
folder_path = "/mnt/external/patients_UCSF/UCSF-PDGM-v3/UCSF-PDGM-0004_nifti"

# Loop through files in the directory
for file_ in os.listdir(folder_path):
    if file_.endswith(".nii.gz"):
        file_path = os.path.join(folder_path, file_)
        print(f"Reorienting: {file_path}")

        # Apply fslreorient2std and overwrite the file
        subprocess.run(["fslreorient2std", file_path, file_path], check=True)
        print(f"Updated: {file_path}")