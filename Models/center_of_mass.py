import os.path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from patient import Patient

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
REMOTE = r"C:\Users\Anna\PycharmProjects\Brain_Imaging\bias_field_correction_samples"
INPUT_MRI = "T1", "T1c", "T2", "FLAIR"

def compute_center_of_mass(number_id):
    patient = Patient(number_id, local=True)
