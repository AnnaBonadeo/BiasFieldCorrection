import os
import random
import string


CONTROL1 = "UCSF-PDGM-"

# Define main test directory
test_main_dir = "test_dataset"
os.makedirs(test_main_dir, exist_ok=True)

# Number of subfolders to create
num_folders = 5
num_random_files = 6  # Number of extra random .nii.gz files per folder

# Function to generate a random filename
def random_filename(extension=".nii.gz"):
    return CONTROL1 + "".join(random.choices(string.ascii_letters + string.digits, k=8)) + extension

# Creating subfolders with .nii.gz files
for i in range(num_folders):
    folder_name = CONTROL1 + f"subject_{i+1}"
    folder_path = os.path.join(test_main_dir, folder_name)
    os.makedirs(folder_path, exist_ok=True)

    # Creating the required files
    required_files = [CONTROL1 + "T1.nii.gz", CONTROL1 + "T2.nii.gz", CONTROL1 + "FLAIR.nii.gz",
                      CONTROL1 + "T1ce.nii.gz", "T1.nii.gz",
                      "T2.nii.gz", "FLAIR.nii.gz", "T1ce.nii.gz",
                      "brain_parenchyma_segmentation.nii.gz",
                      "brain_segmentation.nii.gz", "tumor_segmentation.nii.gz" ]
    for file in required_files:
        with open(os.path.join(folder_path, file), "w") as f:
            f.write("Simulated NIfTI data")  # Placeholder content

    # Creating additional random .nii.gz files
    for _ in range(num_random_files):
        random_file = random_filename()
        with open(os.path.join(folder_path, random_file), "w") as f:
            f.write("Simulated NIfTI data")  # Placeholder content

print(f"Test dataset created at: {os.path.abspath(test_main_dir)}")
