import os
import subprocess
from save_files import CONTROL1, CONTROL_anat, NEW_DIR



for folder in os.listdir(NEW_DIR):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        
        anat_dir = os.path.join(folder_path, "anat")
        seg_dir = os.path.join(folder_path, "seg")
        reg_dir = os.path.join(folder_path, "reg")
        # Ensure 'reg' directory exists
        os.makedirs(reg_dir, exist_ok=True)
        
        # Paths for segmentation masks
        brain_segmentation = os.path.join(seg_dir, f"{folder_name}_brain_segmentation.nii.gz")
        brain_parenchyma_segmentation = os.path.join(seg_dir, f"{folder_name}_brain_parenchyma_segmentation.nii.gz")
        tumor_segmentation = os.path.join(seg_dir, f"{folder_name}_tumor_segmentation.nii.gz")
        
        

        for nii_file in os.listdir(anat_dir):
            if nii_file.endswith(CONTROL_anat):
                input_path = os.path.join(anat_dir, nii_file)
                base_name = nii_file.split('.')[0]  # Extract base name

                # Define the four N4BiasFieldCorrection commands
                commands = [
                    {
                        "output_name": f"{base_name}_N4_brain",
                        "weight": brain_segmentation,
                        "mask": brain_segmentation,
                    },
                    {
                        "output_name": f"{base_name}_N4_brain_parenchyma",
                        "weight": brain_parenchyma_segmentation,
                        "mask": brain_parenchyma_segmentation,
                    },
                    {
                        "output_name": f"{base_name}_N4_parenchyma_brain",
                        "weight": brain_parenchyma_segmentation,
                        "mask": brain_segmentation,
                    },
                    {
                        "output_name": f"{base_name}_N4_parenchyma",
                        "weight": brain_parenchyma_segmentation,
                        "mask": brain_parenchyma_segmentation,
                    }
                ]

                for command in commands:
                    output_corrected = os.path.join(reg_dir, f"{command['output_name']}.nii.gz")
                    output_biasfield = os.path.join(reg_dir, f"biasfield_{command['output_name']}.nii.gz")

                    n4_command = (
                        f"N4BiasFieldCorrection -d 3 "
                        f"-i {input_path} "
                        f"-w {command['weight']} "
                        f"-x {command['mask']} "
                        f"-o [ {output_corrected}, {output_biasfield} ]"
                    )
                    print(f"Executing: {n4_command}")
                    subprocess.run(n4_command, shell=True)
                    # Denoising step
                    denoised_output = os.path.join(reg_dir, f"{command['output_name']}_dn.nii.gz")
                    denoise_command = f"DenoiseImage -d 3 -i {output_corrected} -o {denoised_output}"
                    print(f"Executing: {denoise_command}")
                    subprocess.run(denoise_command, shell=True)




