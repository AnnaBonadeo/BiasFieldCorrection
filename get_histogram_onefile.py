import os
import numpy as np
import matplotlib.pyplot as plt

file_path = "/Users/matte/Desktop/BiasFieldCorrection/bias_field_correction_samples/array/UCSF-PDGM-0371_T2_N4_healthy_mask_rescaled.npy"
file_array = np.load(file_path).astype(np.float32)  # or np.float64

hist, bins = np.histogram(file_array, bins=65536, range=(0, 65536))

plt.hist(hist, bins=bins)
plt.figure(figsize=(10, 6))
plt.plot(bins[1:-1], hist[1:])  # The bin edges exclude the last one
plt.xlabel('Voxel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Voxel Intensities (Rescaled)')
plt.grid(True)
plt.show()