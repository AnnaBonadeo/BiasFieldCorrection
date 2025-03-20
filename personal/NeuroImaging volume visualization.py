#In the context of MRI (Magnetic Resonance Imaging), EPI stands for Echo Planar Imaging.
# It is a fast imaging technique commonly used in functional and diffusion MRI studies.

'''Applications:
Functional MRI (fMRI): EPI is widely used in fMRI to measure changes in blood oxygenation levels over time, allowing researchers to track brain activity.
Diffusion MRI (DWI/DTI): EPI enables diffusion-weighted imaging, which is crucial for mapping white matter tracts and identifying tissue microstructure.
Perfusion Imaging: It can also be used for imaging cerebral blood flow'''

from nilearn import datasets
from nilearn.image.image import mean_img
from nilearn import plotting


# By default, 2nd subject will be fetched
haxby_dataset = datasets.fetch_haxby()

# print basic information on the dataset
print(
    f"First anatomical nifti image (3D) located is at: {haxby_dataset.anat[0]}"
)
print(
    f"First functional nifti image (4D) is located at: {haxby_dataset.func[0]}"
)


# Compute the mean EPI: we do the mean along the axis 3, which is time
func_filename = haxby_dataset.func[0]
mean_haxby = mean_img(func_filename, copy_header=True)

plotting.plot_epi(mean_haxby, colorbar=True, cbar_tick_format="%i")
plotting.show()
