import os
import subprocess
from save_files import CONTROL1, CONTROL_anat, NEW_DIR



for folder in os.listdir(NEW_DIR):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        seg_dir = os.path.join(folder_path, "seg")
        # Ensure 'reg' directory exists
        os.makedirs(seg_dir, exist_ok=True)

        # Preexisting files paths
        brain_segmentation = os.path.join(seg_dir, f"{folder_name}_brain_segmentation.nii.gz")
        tumor_segmentation = os.path.join(seg_dir, f"{folder_name}_tumor_segmentation.nii.gz")

        # New files paths
        tumor_binary = os.path.join(seg_dir, f"{folder_name}_tumor_binary.nii.gz")
        brain_healthy_segmentation = os.path.join(seg_dir, f"{folder_name}_brain_healthy_segmentation.nii.gz")

        command_tumor = f"fslmaths {tumor_segmentation} -bin -nan {tumor_binary}"
        command_brain_healthy = f"fslmaths {brain_segmentation} -sub {tumor_binary} -bin -nan {brain_healthy_segmentation}"

        # Run commands and handle errors
        try:
            print(f"Running command: {command_tumor}")
            subprocess.run(command_tumor, shell=True, check=True)
            print(f"Successfully created tumor binary: {tumor_binary}")
        except subprocess.CalledProcessError as e:
            print(f"Error while creating tumor binary for {folder}: {e}")
            continue

        try:
            print(f"Running command: {command_brain_healthy}")
            subprocess.run(command_brain_healthy, shell=True, check=True)
            print(f"Successfully created brain healthy segmentation: {brain_healthy_segmentation}")
        except subprocess.CalledProcessError as e:
            print(f"Error while creating brain healthy segmentation for {folder}: {e}")
