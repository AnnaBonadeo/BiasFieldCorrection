import os.path
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from concurrent.futures import ProcessPoolExecutor, as_completed
from patient import Patient

def compute_center_of_mass(patient):
