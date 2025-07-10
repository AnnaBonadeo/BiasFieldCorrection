import os
import re
import numpy as np
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor, as_completed

# Adjust this import path as needed
from Models.patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = ["T1", "T1c", "T2", "FLAIR"]

def plot_com_intensities(patient_data_list, modality="T1"):
    corrections = ["n4bb", "n4hh", "n4bh", "n4hb"]
    colors = {"n4bb": "blue", "n4hh": "green", "n4bh": "red", "n4hb": "purple"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for idx, correction in enumerate(corrections):
        x_vals, y_vals = [], []
        for pdata in patient_data_list:
            com_data = pdata.get("com_data", {})
            if modality not in com_data or correction not in com_data[modality]:
                continue

            com = com_data[modality][correction]
            x, y = com["full"]["intensity"], com["masked"]["intensity"]

            if not (np.isnan(x) or np.isnan(y)):
                x_vals.append(x)
                y_vals.append(y)

        ax = axes[idx]
        ax.scatter(x_vals, y_vals, alpha=0.7, color=colors[correction])
        ax.set_title(f"{modality} - {correction}")
        ax.set_xlabel("Full Volume Intensity")
        ax.set_ylabel("Masked Volume Intensity")
        ax.grid(True)

    fig.suptitle(f"COM Intensities for {modality}", fontsize=16)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
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
