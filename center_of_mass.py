import os
import numpy as np
import re
import matplotlib.pyplot as plt
from Models.patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = ["T1", "T1c", "T2", "FLAIR"]

import matplotlib.pyplot as plt
import numpy as np

def plot_com_intensities(patient_data_list, modality="T1"):
    corrections = ["n4bb", "n4hh", "n4bh", "n4hb"]
    colors = {"n4bb": "blue", "n4hh": "green", "n4bh": "red", "n4hb": "purple"}

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()  # to index axes as a simple list

    for idx, correction in enumerate(corrections):
        x_vals = []
        y_vals = []

        for pdata in patient_data_list:
            com_data = pdata.get("com_data", {})
            if modality not in com_data or correction not in com_data[modality]:
                continue

            com = com_data[modality][correction]
            x = com["full"]["intensity"]
            y = com["masked"]["intensity"]

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
        p = Patient(numeric_id, local=False)
        p.compute_center_of_mass()  # no args needed now
        data = {"id": numeric_id, "com_data": p.com_data}
        del p  # free memory
        print(f"Processed patient: {numeric_id}")
        return data
    except Exception as e:
        print(f"Error processing {numeric_id}: {e}")
        return None

# MAIN
folders = os.listdir(NEW_DIR)
patient_data_list = []

for folder in folders:
    pdata = process_patient(folder)
    if pdata is not None:
        patient_data_list.append(pdata)

for modality in INPUT_MRI:
    plot_com_intensities(patient_data_list, modality=modality)
