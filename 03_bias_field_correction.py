# Imports
import sys
import os
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Constants

# Functions


# Get intensity array from MRI image
raw_img_sitk = sitk.ReadImage('bias_field_correction_samples/MR_Gd.nii', sitk.sitkFloat32) # define the pixel type
raw_img_sitk_arr = sitk.GetArrayFromImage(raw_img_sitk) # return the pixel representation of the MRI image

# Delimit regions of interest with Mask
# rescale into readable scale the intensities
transformed = sitk.RescaleIntensity(raw_img_sitk, 0, 255)
transformed_arr = sitk.GetArrayFromImage(transformed)

# Check if data is brain data
print(transformed_arr.shape)  # Should be (Slices, Height, Width)
print("Min:", np.min(transformed_arr), "Max:", np.max(transformed_arr))  # Check intensity values

# Select a middle slice for visualization, central part of the brain
middle_slice = transformed_arr.shape[0] // 2
# transformed_arr[middle_slice] # get the 'y' list and 'z' list

# Create mask



# Pick different slices to check for brain structure - Axial cut
for i in range(10, transformed_arr.shape[0] - 10):
    plt.imshow(transformed_arr[i], cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    plt.axis("off")
    plt.show()

# Pick different slices to check for brain structure - Coronal cut
# The image appears rotated by 180°
for i in range(10, transformed_arr.shape[1] - 10):
    plt.imshow(np.rot90(transformed_arr[:, i], 2), cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    # plt.show()

# Pick different slices to check for brain structure - Sagittal cut
for i in range(10, transformed_arr.shape[2] - 10):
    plt.imshow(np.rot90(transformed_arr[:, :, i],2), cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    # plt.show()

