import multiprocessing
import os
from save_files import CONTROL1, NEW_DIR

def process_folder(folder):
    folder_path = os.path.join(NEW_DIR, folder)
    folder_name = folder.split('_')[0]

    # Check if it's a subject folder
    if os.path.isdir(folder_path) and folder.startswith(CONTROL1):
        print(f"Processing folder: {folder}")

        array_dir = os.path.join(folder_path, "array")
        histograms_dir = os.path.join(folder_path, "histograms")

        # Ensure directories exist
        os.makedirs(array_dir, exist_ok=True)
        os.makedirs(histograms_dir, exist_ok=True)

        for nii_file in os.listdir(array_dir):


# MAIN PIPELINE
folders = os.listdir(NEW_DIR)

with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
    pool.map(process_folder, folders)