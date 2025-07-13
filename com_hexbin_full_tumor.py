import os
import re
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Adjust this import path as needed
from Models.patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = ["T1", "T1c", "T2", "FLAIR"]

import matplotlib.pyplot as plt
import numpy as np

import matplotlib.pyplot as plt
import numpy as np

def plot_com_from_hexbins_with_tumor(patient_data_list, modality="T1"):
    corrections = ["n4bb", "n4hh", "n4bh", "n4hb"]
    colors = {"n4bb": "blue", "n4hh": "green", "n4bh": "red", "n4hb": "purple"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, correction in enumerate(corrections):
        x_full, y_full = [], []
        x_tumor, y_tumor = [], []

        for pdata in patient_data_list:
            com_data = pdata.get("com_data", {})
            if modality not in com_data or correction not in com_data[modality]:
                continue

            com = com_data[modality][correction]
            com_full = com.get("com_full", None)
            com_tumor = com.get("com_tumor", None)

            if com_full is not None and not (np.isnan(com_full[0]) or np.isnan(com_full[1])):
                x_full.append(com_full[0])
                y_full.append(com_full[1])

            if com_tumor is not None and not (np.isnan(com_tumor[0]) or np.isnan(com_tumor[1])):
                x_tumor.append(com_tumor[0])
                y_tumor.append(com_tumor[1])

        ax = axes[idx]
        ax.scatter(x_full, y_full, alpha=0.7, color=colors[correction], label="Full Volume", marker='o')
        ax.scatter(x_tumor, y_tumor, alpha=0.7, color=colors[correction], label="Tumor Region", marker='x')

        ax.set_title(f"{modality} - {correction}")
        ax.set_xlabel("Weighted CoM: Image Intensity")
        ax.set_ylabel("Weighted CoM: Bias Field Intensity")
        ax.grid(True)
        ax.legend()

    plt.tight_layout()
    plt.show()



def process_patient(folder):
    full_path = os.path.join(NEW_DIR, folder)
    if not os.path.isdir(full_path):
        return None

    match = re.search(r'\d+', folder)
    if not match:
        return None

    numeric_id = match.group()
    try:
        p = Patient(numeric_id, local=False, mmap=True)  # Custom `mmap` arg if supported
        p.compute_center_of_mass()
        data = {"id": numeric_id, "com_data": p.com_data}
        del p  # Explicitly free memory
        print(f"Processed patient: {numeric_id}")
        return data
    except Exception as e:
        print(f"Error processing {numeric_id}: {e}")
        return None


def parallel_process_patients(folders, max_workers=4):
    patient_data_list = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_patient, folder): folder for folder in folders}
        for future in as_completed(futures):
            try:
                result = future.result()
                if result is not None:
                    patient_data_list.append(result)
            except Exception as e:
                folder = futures[future]
                print(f"Exception processing folder {folder}: {e}")
    return patient_data_list


# MAIN
if __name__ == "__main__":
    folders = os.listdir(NEW_DIR)
    folders = [f for f in folders if os.path.isdir(os.path.join(NEW_DIR, f))]

    print("Starting parallel processing...")
    patient_data_list = parallel_process_patients(folders, max_workers=4)  # Adjust as needed

    for modality in INPUT_MRI:
        plot_com_intensities(patient_data_list, modality=modality)
