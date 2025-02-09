import os
import subprocess

from save_files import CONTROL1

MAIN_DIR = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\try_directory"

for folder in os.listdir(MAIN_DIR):
    folder_path = os.path.join(MAIN_DIR, folder)

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        for file_ in os.listdir(folder_path):
            if file_.endswith(".nii.gz"):
                file_path = os.path.join(folder_path, file_)
                print(f"Reorienting: {file_path}")

                # Apply fslreorient2std and overwrite the file
                subprocess.run(["fslreorient2std", file_path, file_path], check=True)
                print(f"Updated: {file_path}")


