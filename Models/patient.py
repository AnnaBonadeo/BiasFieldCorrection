import os.path
import numpy as np
import pandas as pd
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
REMOTE = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

class Patient():
    def __init__(self, id_, local=False, mmap=False):
        self.id = id_
        self.prefix = f"UCSF-PDGM-{self.id}"
        path = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples" if local else "/mnt/external/reorg_patients_UCSF"
        self.dir = f"{path}/{self.prefix}_nifti"
        #T1
        self.native_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_rescaled.npy"))
        self.n4hh_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_healthy_mask_rescaled.npy"))
        self.n4bb_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_brain_rescaled.npy"))
        self.n4bh_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_brain_healthy_mask_rescaled.npy"))
        self.n4hb_t1_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1_N4_healthy_mask_brain_rescaled.npy"))
        #Biasfield T1
        self.biasfield_n4hh_t1_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1_N4_healthy_mask_rescaled.npy"))
        self.biasfield_n4bb_t1_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1_N4_brain_rescaled.npy"))
        self.biasfield_n4bh_t1_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1_N4_brain_healthy_mask_rescaled.npy"))
        self.biasfield_n4hb_t1_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1_N4_healthy_mask_brain_rescaled.npy"))

        #T2
        self.native_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_rescaled.npy"))
        self.n4hh_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_healthy_mask_rescaled.npy"))
        self.n4bb_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_brain_rescaled.npy"))
        self.n4bh_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_brain_healthy_mask_rescaled.npy"))
        self.n4hb_t2_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T2_N4_healthy_mask_brain_rescaled.npy"))
        #Biasfield T2
        self.biasfield_n4hh_t2_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T2_N4_healthy_mask_rescaled.npy"))
        self.biasfield_n4bb_t2_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T2_N4_brain_rescaled.npy"))
        self.biasfield_n4bh_t2_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T2_N4_brain_healthy_mask_rescaled.npy"))
        self.biasfield_n4hb_t2_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T2_N4_healthy_mask_brain_rescaled.npy"))

        #T1c
        self.native_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_rescaled.npy"))
        self.n4hh_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_healthy_mask_rescaled.npy"))
        self.n4bb_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_brain_rescaled.npy"))
        self.n4bh_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_brain_healthy_mask_rescaled.npy"))
        self.n4hb_t1c_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_T1c_N4_healthy_mask_brain_rescaled.npy"))
        #Biasfield T1c
        self.biasfield_n4hh_t1c_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1c_N4_healthy_mask_rescaled.npy"))
        self.biasfield_n4bb_t1c_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1c_N4_brain_rescaled.npy"))
        self.biasfield_n4bh_t1c_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1c_N4_brain_healthy_mask_rescaled.npy"))
        self.biasfield_n4hb_t1c_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_T1c_N4_healthy_mask_brain_rescaled.npy"))
        #FLAIR
        self.native_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_rescaled.npy"))
        self.n4hh_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_healthy_mask_rescaled.npy"))
        self.n4bb_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_brain_rescaled.npy"))
        self.n4bh_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_brain_healthy_mask_rescaled.npy"))
        self.n4hb_flair_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_FLAIR_N4_healthy_mask_brain_rescaled.npy"))
        #Biasfield flair
        self.biasfield_n4hh_flair_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_FLAIR_N4_healthy_mask_rescaled.npy"))
        self.biasfield_n4bb_flair_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_FLAIR_N4_brain_rescaled.npy"))
        self.biasfield_n4bh_flair_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_FLAIR_N4_brain_healthy_mask_rescaled.npy"))
        self.biasfield_n4hb_flair_array = np.load(os.path.join(self.dir, f"array/biasfield_{self.prefix}_FLAIR_N4_healthy_mask_brain_rescaled.npy"))
        #TUMOR BINARY
        self.tumor_binary_array = np.load(os.path.join(self.dir, f"array/{self.prefix}_tumor_binary_array.npy"))

    def _calculate_median_distances(self, native, n4bb, n4hh, n4bh, n4hb):

        tumor_binary_array = self.tumor_binary_array
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
        t1_medians = []
        for i in self._calculate_median_distances(self.native_t1_array, self.n4bb_t1_array, self.n4hh_t1_array, self.n4bh_t1_array, self.n4hb_t1_array):
            t1_medians.append(i)
        return t1_medians
    def get_median_distance_T2(self):
        t2_medians = []
        for i in self._calculate_median_distances(self.native_t2_array, self.n4bb_t2_array, self.n4hh_t2_array, self.n4bh_t2_array, self.n4hb_t2_array):
            t2_medians.append(i)
        return t2_medians
    def get_median_distance_T1c(self):
        t1c_medians = []
        for i in self._calculate_median_distances(self.native_t1c_array, self.n4bb_t1c_array, self.n4hh_t1c_array, self.n4bh_t1c_array, self.n4hb_t1c_array):
            t1c_medians.append(i)
        return t1c_medians
    def get_median_distance_FLAIR(self):
        flair_medians = []
        for i in self._calculate_median_distances(self.native_flair_array, self.n4bb_flair_array, self.n4hh_flair_array, self.n4bh_flair_array,
                                                  self.n4hb_flair_array):
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
    def _center_and_intensity(self, volume):
        # Computations for full volume
        total_mass_volume = np.sum(volume)
        z, y, x = np.indices(volume.shape)
        if total_mass_volume == 0:
            coords_full = None
            intensity_at_com_full = np.nan
        else:
            z_com_full = np.sum(z * volume) / total_mass_volume
            y_com_full = np.sum(y * volume) / total_mass_volume
            x_com_full = np.sum(x * volume) / total_mass_volume
            coords_full = [[z_com_full], [y_com_full], [x_com_full]]
            intensity_at_com_full = map_coordinates(volume, coords_full, order=1)[0]

        # Computations for masked volume
        mask = self.tumor_binary_array
        volume_masked = volume * mask
        total_mass_masked = np.sum(volume_masked)
        if total_mass_masked == 0:
            coords_masked = None
            intensity_at_com_masked = np.nan
        else:
            z_com_masked = np.sum(z * volume_masked) / total_mass_masked
            y_com_masked = np.sum(y * volume_masked) / total_mass_masked
            x_com_masked = np.sum(x * volume_masked) / total_mass_masked
            coords_masked = [[z_com_masked], [y_com_masked], [x_com_masked]]
            intensity_at_com_masked = map_coordinates(volume_masked, coords_masked, order=1)[0]

        return {
            "full": {"coords": coords_full, "intensity": intensity_at_com_full},
            "masked": {"coords": coords_masked, "intensity": intensity_at_com_masked},
            }
    def compute_center_of_mass(self):
            self.com_data = {
                "T1": {
                    "n4bb": self._center_and_intensity(self.n4bb_t1_array),
                    "n4hh": self._center_and_intensity(self.n4hh_t1_array),
                    "n4bh": self._center_and_intensity(self.n4bh_t1_array),
                    "n4hb": self._center_and_intensity(self.n4hb_t1_array),
                },
                "T1c": {
                    "n4bb": self._center_and_intensity(self.n4bb_t1c_array),
                    "n4hh": self._center_and_intensity(self.n4hh_t1c_array),
                    "n4bh": self._center_and_intensity(self.n4bh_t1c_array),
                    "n4hb": self._center_and_intensity(self.n4hb_t1c_array),
                },
                "T2": {
                    "n4bb": self._center_and_intensity(self.n4bb_t2_array),
                    "n4hh": self._center_and_intensity(self.n4hh_t2_array),
                    "n4bh": self._center_and_intensity(self.n4bh_t2_array),
                    "n4hb": self._center_and_intensity(self.n4hb_t2_array),
                },
                "FLAIR": {
                    "n4bb": self._center_and_intensity(self.n4bb_flair_array),
                    "n4hh": self._center_and_intensity(self.n4hh_flair_array),
                    "n4bh": self._center_and_intensity(self.n4bh_flair_array),
                    "n4hb": self._center_and_intensity(self.n4hb_flair_array),
                },
            }
            return self.com_data

    def _com_hexbin(self, volume, biasfield_volume):
        mask = volume > 0
        tumor_mask_array = self.tumor_binary_array

        if tumor_mask_array is not None:
            tumor_mask = (tumor_mask_array > 0) & mask
        else:
            tumor_mask = np.zeros_like(volume, dtype=bool)

        x_vals_full = volume[mask]
        y_vals_full = biasfield_volume[mask]

        x_vals_tumor = volume[tumor_mask]
        y_vals_tumor = biasfield_volume[tumor_mask]

        # Center of mass for full image (weighted by hexbin)
        fig1, ax1 = plt.subplots()
        hb_full = ax1.hexbin(x_vals_full, y_vals_full, gridsize=50, cmap='Blues', mincnt=1)
        counts_full = hb_full.get_array() #bin counts (weights): how many point in the bin?
        offsets_full = hb_full.get_offsets() #bin centers

        plt.close(fig1)

        if len(counts_full) > 0:
            com_full = (
                np.average(offsets_full[:, 0], weights=counts_full),
                np.average(offsets_full[:, 1], weights=counts_full)
            )
        else:
            com_full = (None, None)

        # Center of mass for tumor region (weighted by hexbin)
        if len(x_vals_tumor) > 0:
            fig2, ax2 = plt.subplots()
            hb_tumor = ax2.hexbin(x_vals_tumor, y_vals_tumor, gridsize=50, cmap='Reds', mincnt=1)
            counts_tumor = hb_tumor.get_array()
            offsets_tumor = hb_tumor.get_offsets()
            plt.close(fig2)

            if len(counts_tumor) > 0:
                com_tumor = (
                    np.average(offsets_tumor[:, 0], weights=counts_tumor),
                    np.average(offsets_tumor[:, 1], weights=counts_tumor)
                )
            else:
                com_tumor = (None, None)
        else:
            com_tumor = (None, None)

        return {
            "com_full": com_full,
            "com_tumor": com_tumor
        }

    def compute_com_scatterplot(self):
        self.com_data = {
            "T1": {
                "n4bb": self._com_hexbin(self.n4bb_t1_array, self.biasfield_n4bb_t1_array),
                "n4hh": self._com_hexbin(self.n4hh_t1_array, self.biasfield_n4hh_t1_array),
                "n4bh": self._com_hexbin(self.n4bh_t1_array, self.biasfield_n4bh_t1_array),
                "n4hb": self._com_hexbin(self.n4hb_t1_array, self.biasfield_n4hb_t1_array),
            },
            "T1c": {
                "n4bb": self._com_hexbin(self.n4bb_t1c_array, self.biasfield_n4bb_t1c_array),
                "n4hh": self._com_hexbin(self.n4hh_t1c_array, self.biasfield_n4hh_t1c_array),
                "n4bh": self._com_hexbin(self.n4bh_t1c_array, self.biasfield_n4bh_t1c_array ),
                "n4hb": self._com_hexbin(self.n4hb_t1c_array, self.biasfield_n4hb_t1c_array),
            },
            "T2": {
                "n4bb": self._com_hexbin(self.n4bb_t2_array, self.biasfield_n4bb_t2_array),
                "n4hh": self._com_hexbin(self.n4hh_t2_array, self.biasfield_n4hh_t2_array),
                "n4bh": self._com_hexbin(self.n4bh_t2_array, self.biasfield_n4bh_t2_array),
                "n4hb": self._com_hexbin(self.n4hb_t2_array, self.biasfield_n4hb_t2_array),
            },
            "FLAIR": {
                "n4bb": self._com_hexbin(self.n4bb_flair_array, self.biasfield_n4bb_flair_array),
                "n4hh": self._com_hexbin(self.n4hh_flair_array, self.biasfield_n4hh_flair_array),
                "n4bh": self._com_hexbin(self.n4bh_flair_array, self.biasfield_n4bh_flair_array),
                "n4hb": self._com_hexbin(self.n4hb_flair_array, self.biasfield_n4hb_flair_array),
            },
        }
        return self.com_data