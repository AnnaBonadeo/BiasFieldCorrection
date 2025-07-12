import os.path
import pandas as pd
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from Models.patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

def process_patient(folder):
    full_path = os.path.join(NEW_DIR, folder)
    if not os.path.isdir(full_path):
        return None

    match = re.search(r'\d+', folder)
    if match:
        numeric_id = match.group()
        try:
            p = Patient(numeric_id, local=False)
            df = p.get_patient_df()
            print(f"Processing patient: {numeric_id}")
            return df
        except Exception as e:
            print(f"Error processing {numeric_id}: {e}")
            return None
    return None

folders = os.listdir(NEW_DIR)
all_dfs = []

with ProcessPoolExecutor() as executor:
    futures = [executor.submit(process_patient, folder) for folder in folders]
    for future in as_completed(futures):
        result = future.result()
        if result is not None:
            all_dfs.append(result)

all_dfs = pd.concat(all_dfs, ignore_index=True)
all_dfs_long = pd.melt(
    all_dfs,
    id_vars=["Patient", "Modality"],
    value_vars=["Native", "N4_Brain", "N4_Healthy", "N4_Brain_Healthy", "N4_Healthy_Brain"],
    var_name="method",
    value_name="median_distance"
)
