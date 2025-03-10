import numpy as np
import nibabel as nib

# Define image size (simulating an MRI scan with some intensity variations)
image_shape = (128, 128, 64)  # Typical 3D MRI size

# Generate a synthetic image with intensity variations (simulating bias field effect)
np.random.seed(42)
base_image = np.zeros(image_shape, dtype=np.float32)

# Simulating a simple bias field (low-frequency intensity variation)
x, y, z = np.meshgrid(np.linspace(-1, 1, image_shape[0]),
                       np.linspace(-1, 1, image_shape[1]),
                       np.linspace(-1, 1, image_shape[2]), indexing="ij")

bias_field = 1 + 0.2 * (x**2 + y**2 - z**2)
base_image += bias_field * 100  # Scale intensity

# Add some Gaussian noise to simulate real MRI noise
noise = np.random.normal(0, 5, image_shape).astype(np.float32)
base_image += noise

# Ensure intensity values are in a realistic range
base_image = np.clip(base_image, 0, 255)

# Convert to NIfTI format
nifti_image = nib.Nifti1Image(base_image, affine=np.eye(4))

# Save as NIfTI file
nifti_filename = "bias_field_correction_samples/UCSF-PDGM-0013_blabla/reg/synthetic_mri.nii.gz"
nib.save(nifti_image, nifti_filename)

print(f"Generated NIfTI file: {nifti_filename}")
