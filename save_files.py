import os
import shutil

# Constants
CONTROL1 = "UCSF-PDGM-"

# File groups
CONTROL_anat = ("T1.nii.gz", "T1ce.nii.gz", "T2.nii.gz", "FLAIR.nii.gz")
CONTROL_seg = ("brain_parenchyma_segmentation.nii.gz", "brain_segmentation.nii.gz", "tumor_segmentation.nii.gz")

# Paths
MAIN_DIR = "test_dataset"
NEW_DIR = "try_directory"

# Iterate over each subject folder
for folder in os.listdir(MAIN_DIR):
    folder_path = os.path.join(MAIN_DIR, folder)

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        # Create new subject directory structure
        subject_new_path = os.path.join(NEW_DIR, folder)
        anat_dir = os.path.join(subject_new_path, "anat")
        seg_dir = os.path.join(subject_new_path, "seg")
        reg_dir = os.path.join(subject_new_path, "reg")

        os.makedirs(anat_dir, exist_ok=True)
        os.makedirs(seg_dir, exist_ok=True)
        os.makedirs(reg_dir, exist_ok=True)  # This remains empty

        # Move files into appropriate directories by copying them
        # If we don't want to keep the prior copy, use os.rename(source_path, dest_path)
        # DISCLAIMER! os.rename() cannot move files among different disks so i'm not sure
        # it would work in our case
        for file_ in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file_)
            if file_.startswith(CONTROL1):
                if file_.endswith(CONTROL_anat):
                    shutil.copy2(file_path, os.path.join(anat_dir, file_))
                    print(f"Copied {file_} to {anat_dir}")

                elif file_.endswith(CONTROL_seg):
                    shutil.copy2(file_path, os.path.join(seg_dir, file_))
                    print(f"Copied {file_} to {seg_dir}")

print("Processing complete.")
