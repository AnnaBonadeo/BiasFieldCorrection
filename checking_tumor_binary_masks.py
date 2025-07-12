import os
import subprocess
from save_files import NEW_DIR

empty_tumor_patients = []

for folder in os.listdir(NEW_DIR):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    if os.path.isdir(folder_path):
        seg_dir = os.path.join(folder_path, "seg")
        tumor_binary = os.path.join(seg_dir, f"{folder_name}_tumor_binary.nii.gz")

        if os.path.exists(tumor_binary):
            try:
                result = subprocess.run(["fslstats", tumor_binary, "-V"], capture_output=True, text=True, check=True)
                voxel_count = int(result.stdout.strip().split()[0])
                if voxel_count == 0:
                    empty_tumor_patients.append(folder)
            except subprocess.CalledProcessError as e:
                print(f"Error reading stats for {tumor_binary}: {e}")

print(f"\nTotal patients with empty tumor masks: {len(empty_tumor_patients)}")
if empty_tumor_patients:
    print("Patients with empty tumor masks:")
    for patient in empty_tumor_patients:
        print(patient)
