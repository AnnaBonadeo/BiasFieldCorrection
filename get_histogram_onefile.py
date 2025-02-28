import os
import numpy as np
import matplotlib.pyplot as plt

file_path = input("Enter the file path: ")
file_array = np.load(file_path).astype(np.float32)  # or np.float64

hist, bins = np.histogram(file_array, bins=65536, range=(0, 65536))

plt.hist(hist, bins=bins)
plt.figure(figsize=(10, 6))
plt.plot(bins[:-1], hist)  # The bin edges exclude the last one
plt.xlabel('Voxel Intensity')
plt.ylabel('Frequency')
plt.title('Histogram of Voxel Intensities (Rescaled)')
plt.grid(True)
plt.show()