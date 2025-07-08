import os.path
import numpy as np
import pandas as pd
import re

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
REMOTE = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples"

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
        median_whole_brain_native = np.median(native)
        median_whole_brain_n4bb = np.median(n4bb)
        median_whole_brain_n4hh = np.median(n4hh)
        median_whole_brain_n4bh = np.median(n4bh)
        median_whole_brain_n4hb = np.median(n4hb)
        median_tumor_native = np.median(native_tumor_array)
        median_tumor_n4bb = np.median(n4bb_tumor_array)
        median_tumor_n4hh = np.median(n4hh_tumor_array)
        median_tumor_n4bh = np.median(n4bh_tumor_array)
        median_tumor_n4hb = np.median(n4hb_tumor_array)
        median_distance_native = median_whole_brain_native - median_tumor_native
        median_distance_n4bb = median_whole_brain_n4bb - median_tumor_n4bb
        median_distance_n4hh = median_whole_brain_n4hh - median_tumor_n4hh
        median_distance_n4bh = median_whole_brain_n4bh - median_tumor_n4bh
        median_distance_n4hb = median_whole_brain_n4hb - median_tumor_n4hb
        return [float(median_distance_native), float(median_distance_n4bb), float(median_distance_n4hh), float(median_distance_n4bh), float(median_distance_n4hb)]

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


if __name__ == "__main__":
     dir = NEW_DIR
     all_dfs = []
     for folder in os.listdir(dir):
         full_path = os.path.join(dir, folder)
         if os.path.isdir(full_path):
             # Extract numeric part from folder name (e.g., 'UCSF-PDGM-0455_nifti' -> '0455')
             match = re.search(r'\d+', folder)
             if match:
                 numeric_id = match.group()
                 try:
                     p = Patient(numeric_id, local=True) # change local = False for remote
                     df = p.get_patient_df()
                     all_dfs.append(df)
                 except Exception as e:
                    print(f"Error processing {numeric_id}: {e}")
     all_dfs = pd.concat(all_dfs, ignore_index=True)
     all_dfs.to_csv(f"{dir}/00_patient_df.csv", index=False)