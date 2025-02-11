# Write a code to save in new directories only the following files:
# t1,t1c,t2,flair,brain_segmentation, brain_parenchyma, tumor_segmentation
import os

# Constants
#CONTROL = "UCSF-PDGM-"
CONTROL2 = "T1.nii.gz", "T1ce.nii.gz", "T2.nii.gz", "FLAIR.nii.gz"
NEW_DIR = "C:/Users/Anna/PycharmProjects/Brain_Imaging/try_directory"

main_dir = "C:/Users/Anna/PycharmProjects/Brain_Imaging/test_dataset"

for folder in os.listdir(main_dir):
    folder_path = os.path.join(main_dir, folder)

    if os.path.isdir(folder_path):
        for file_ in os.listdir(folder_path):
            if file_.endswith(CONTROL2):
                file_path = os.path.join(folder_path, file_)
                new_dir_path = os.path.join(NEW_DIR, folder)
                os.makedirs(new_dir_path, exist_ok=True)
                new_file_path = os.path.join(new_dir_path, file_)
                os.rename(file_path, new_file_path)








