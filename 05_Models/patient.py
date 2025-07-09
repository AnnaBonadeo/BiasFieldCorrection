import os.path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed


NEW_DIR = "/mnt/external/reorg_patients_UCSF"
REMOTE = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

class Patient():
    def __init__(self, id_, local=False):
        self.id = id_
        self.prefix = f"UCSF-PDGM-{self.id}"
        path = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples" if local else "/mnt/external/reorg_patients_UCSF"
        self.dir = f"{path}/{self.prefix}_nifti"

    def _calculate(self, native, n4bb, n4hh, n4bh, n4hb):

        tumor_binary_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_tumor_binary_array.npy"))
        native_tumor_array = native * tumor_binary_array
        n4hh_tumor_array = n4hh * tumor_binary_array
        n4bb_tumor_array = n4bb * tumor_binary_array
        n4hb_tumor_array = n4hb * tumor_binary_array
        n4bh_tumor_array = n4bh * tumor_binary_array

        median_whole_brain_native = np.median(native[native > 0])
        median_whole_brain_n4bb = np.median(n4bb[n4bb > 0])
        median_whole_brain_n4hh = np.median(n4hh[n4hh > 0])
        median_whole_brain_n4bh = np.median(n4bh[n4bh > 0])
        median_whole_brain_n4hb = np.median(n4hb[n4hb > 0])

        median_tumor_native = np.median(native_tumor_array[native_tumor_array > 0])
        median_tumor_n4bb = np.median(n4bb_tumor_array[n4bb_tumor_array > 0])
        median_tumor_n4hh = np.median(n4hh_tumor_array[n4hh_tumor_array > 0])
        median_tumor_n4bh = np.median(n4bh_tumor_array[n4bh_tumor_array > 0])
        median_tumor_n4hb = np.median(n4hb_tumor_array[n4hb_tumor_array > 0])

        median_distance_native = median_whole_brain_native - median_tumor_native
        median_distance_n4bb = median_whole_brain_n4bb - median_tumor_n4bb
        median_distance_n4hh = median_whole_brain_n4hh - median_tumor_n4hh
        median_distance_n4bh = median_whole_brain_n4bh - median_tumor_n4bh
        median_distance_n4hb = median_whole_brain_n4hb - median_tumor_n4hb

        return [
            float(median_distance_native),
            float(median_distance_n4bb),
            float(median_distance_n4hh),
            float(median_distance_n4bh),
            float(median_distance_n4hb)
        ]

    def get_median_distance_T1(self):
        # Load
        native_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_rescaled.npy"))
        n4hh_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_healthy_mask_rescaled.npy"))
        n4bb_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_brain_rescaled.npy"))
        n4bh_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_brain_healthy_mask_rescaled.npy"))
        n4hb_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_healthy_mask_brain_rescaled.npy"))
        t1_medians = []
        for i in self._calculate(native_t1_array, n4bb_t1_array, n4hh_t1_array, n4bh_t1_array, n4hb_t1_array):
            t1_medians.append(i)
        return t1_medians

    def get_median_distance_T2(self):
        native_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_rescaled.npy"))
        n4hh_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_healthy_mask_rescaled.npy"))
        n4bb_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_brain_rescaled.npy"))
        n4bh_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_brain_healthy_mask_rescaled.npy"))
        n4hb_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_healthy_mask_brain_rescaled.npy"))
        t2_medians = []
        for i in self._calculate(native_t2_array, n4bb_t2_array, n4hh_t2_array, n4bh_t2_array, n4hb_t2_array):
            t2_medians.append(i)
        return t2_medians
    def get_median_distance_T1c(self):
        native_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_rescaled.npy"))
        n4hh_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_healthy_mask_rescaled.npy"))
        n4bb_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_brain_rescaled.npy"))
        n4bh_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_brain_healthy_mask_rescaled.npy"))
        n4hb_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_healthy_mask_brain_rescaled.npy"))
        t1c_medians = []
        for i in self._calculate(native_t1c_array, n4bb_t1c_array, n4hh_t1c_array, n4bh_t1c_array, n4hb_t1c_array):
            t1c_medians.append(i)
        return t1c_medians
    def get_median_distance_FLAIR(self):
        native_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_rescaled.npy"))
        n4hh_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_healthy_mask_rescaled.npy"))
        n4bb_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_brain_rescaled.npy"))
        n4bh_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_brain_healthy_mask_rescaled.npy"))
        n4hb_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_healthy_mask_brain_rescaled.npy"))
        flair_medians = []
        for i in self._calculate(native_flair_array, n4bb_flair_array, n4hh_flair_array, n4bh_flair_array,
                                 n4hb_flair_array):
            flair_medians.append(i)
        return flair_medians
    def get_patient_df(self):
        self.t1 = self.get_median_distance_T1()
        self.t2 = self.get_median_distance_T2()
        self.t1c = self.get_median_distance_T1c()
        self.flair = self.get_median_distance_FLAIR()
        df = pd.DataFrame()
        data = []
        for modality, values in zip(["T1", "T2", "T1c", "FLAIR"],[self.t1, self.t2, self.t1c, self.flair]):
            data.append({"Patient":self.id,
                         "Modality":modality,
                         'Native': values[0],
                         'N4_Brain': values[1],
                         'N4_Healthy': values[2],
                         'N4_Brain_Healthy': values[3],
                         'N4_Healthy_Brain': values[4]
             })
        df = pd.DataFrame(data)
        return df

def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image for medians visualization: ").strip().lower()

    # Normalize input list to lowercase for comparison
    valid_answers = {mri.lower(): mri for mri in INPUT_MRI}

    while user_ans not in valid_answers:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").strip().lower()

    return valid_answers[user_ans]  # Return the correctly formatted value (T1/T1c/T2/FLAIR)

def plot_violin_by_method(df, modality):
    # Filter only the selected modality
    df_modality = df[df["Modality"] == modality]

    # Set up the plot
    plt.figure(figsize=(10, 6))
    sns.violinplot(data=df_modality, x="method", y="median_distance", palette="Set2")

    # Customization
    plt.title(f"Median Distance for {modality} across Patients")
    plt.xlabel("Bias Correction Method")
    plt.ylabel("Median Intensity Distance")
    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

def process_patient(folder):
    full_path = os.path.join(NEW_DIR, folder)
    if not os.path.isdir(full_path):
        return None

    match = re.search(r'\d+', folder)
    if match:
        numeric_id = match.group()
        try:
            p = Patient(numeric_id, local=False) # local = True for local
            df = p.get_patient_df()
            print(f"Processing patient:{numeric_id}")
            return df
        except Exception as e:
            print(f"Error processing {numeric_id}: {e}")
            return None
    return None

if __name__ == "__main__":
    folders = os.listdir(NEW_DIR)
    all_dfs = []

    with ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_patient, folder) for folder in folders]

        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                all_dfs.append(result)

    all_dfs = pd.concat(all_dfs, ignore_index=True)
    all_dfs.to_csv(f"{NEW_DIR}/00_patient_df.csv", index=False)
    all_dfs_long = pd.melt(
        all_dfs,
        id_vars=["Patient", "Modality"],
        value_vars=["Native", "N4_Brain", "N4_Healthy", "N4_Brain_Healthy", "N4_Healthy_Brain"],
        var_name="method",
        value_name="median_distance"
    )

    user_answer_modality = get_user_answer(INPUT_MRI)

    user_answer_all = input("Do you want to plot violin plots for ALL modalities as well? (y/n): ").strip().lower()

    if user_answer_all == 'y':
        unique_modalities = all_dfs_long["Modality"].unique()
        for modality in unique_modalities:
            print(f"\nPlotting for modality: {modality}")
            plot_violin_by_method(all_dfs_long, modality)
    else:
        plot_violin_by_method(all_dfs_long, user_answer_modality)


