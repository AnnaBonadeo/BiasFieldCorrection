# Imports
import SimpleITK as sitk
import numpy as np
import matplotlib.pyplot as plt

# Prova

# Get intensity array from MRI image
raw_img_sitk = sitk.ReadImage('bias_field_correction_samples/MR_Gd.nii', sitk.sitkFloat32) # define the pixel type

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

# Create mask, using LiThreshold for the head mask
head_mask = sitk.LiThreshold(transformed,0,1)

shrinkFactor = 4 # shrink to reduce computational cost
inputImage = raw_img_sitk

inputImage = sitk.Shrink( raw_img_sitk, [ shrinkFactor ] * inputImage.GetDimension() )
maskImage = sitk.Shrink( head_mask, [ shrinkFactor ] * inputImage.GetDimension() )

bias_corrector = sitk.N4BiasFieldCorrectionImageFilter()
corrected = bias_corrector.Execute(inputImage, maskImage)

# Back to original resolution
log_bias_field = bias_corrector.GetLogBiasFieldAsImage(raw_img_sitk)
corrected_image_full_resolution = raw_img_sitk / sitk.Exp( log_bias_field )
bias_field = sitk.Exp( log_bias_field )
bias_field_arr = sitk.GetArrayFromImage(bias_field)

# Plot the new corrected image at full resolution
transformed_corrected = sitk.RescaleIntensity(corrected_image_full_resolution, 0, 255)
transformed_arr_corrected = sitk.GetArrayFromImage(transformed_corrected)

# Axial cut
for i in range(10, transformed_arr_corrected.shape[0] - 10):
    plt.subplot(1, 2, 1)
    plt.imshow(transformed_arr_corrected[i], cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(bias_field_arr[i], cmap="gray")
    plt.title(f"Bias Field - Slice {i}")
    plt.axis("off")

    plt.show()

# Coronal cut
for i in range(10, transformed_arr.shape[1] - 10):
    plt.subplot(1, 2, 1)
    plt.imshow(np.rot90(transformed_arr[:, i], 2), cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(np.rot90(bias_field_arr[:, i], 2), cmap="gray")
    plt.title(f"Bias Field - Slice {i}")
    plt.axis("off")

    # plt.show()

# Sagittal cut
for i in range(10, transformed_arr.shape[2] - 10):
    plt.subplot(1, 2, 1)
    plt.imshow(np.rot90(transformed_arr[:, :, i],2), cmap="gray")
    plt.title(f"Brain MRI - Slice {i}")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(np.rot90(bias_field_arr[:, :, i],2), cmap="gray")
    plt.title(f"Bias Field - Slice {i}")
    plt.axis("off")

    # plt.show()

