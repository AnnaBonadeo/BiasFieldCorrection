from nilearn import datasets
from nilearn import plotting

#Load data!
stat_img = datasets.load_sample_motor_activation_image()
# returns string path to data
print(stat_img)
plotting.plot_glass_brain(stat_img, threshold=3)
plotting.plot_glass_brain(
    stat_img,
    title="plot_glass_brain",
    black_bg=True,
    display_mode="ortho",
    threshold=3,
)

#Print image
plotting.show()

display = plotting.plot_stat_map(stat_img)
#display.close()
