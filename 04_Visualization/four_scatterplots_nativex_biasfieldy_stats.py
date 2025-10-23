import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import scikit_posthocs as sp
import pandas as pd
import concurrent.futures

NEW_DIR = "/mnt/external/reorg_patients_UCSF"
CONTROL1 = "UCSF-PDGM-"
MRI_TYPE = ["T1", "T1c", "T2", "FLAIR"]
BIASFIELD = ["N4BB", "N4HH", "N4BH", "N4HB"]

# ---------------------- User Input ----------------------
def get_user_answer(INPUT_MRI):
    user_ans = input("Enter the MRI image to analyze: ").strip().lower()
    valid_answers = {mri.lower(): mri for mri in INPUT_MRI}
    while user_ans not in valid_answers:
        print("Invalid input. Please try again.")
        user_ans = input("Enter the MRI image to analyze: ").strip().lower()
    return valid_answers[user_ans]

# ---------------------- CoM Calculation ----------------------
def compute_center_of_mass_regions(array_mri, array_biasfield, brain_seg_array, tumor_mask_array, bins=100):
    valid_mask = array_mri > 0
    tumor_mask = (tumor_mask_array > 0) & valid_mask
    brain_only_mask = (brain_seg_array > 0) & (~tumor_mask) & valid_mask

    def compute_center_of_mass(x, y, bins=100):
        if len(x) == 0 or len(y) == 0:
            return (np.nan, np.nan)
        hist, xedges, yedges = np.histogram2d(x, y, bins=bins)
        xcenters = 0.5 * (xedges[:-1] + xedges[1:])
        ycenters = 0.5 * (yedges[:-1] + yedges[1:])
        X, Y = np.meshgrid(xcenters, ycenters, indexing='ij')
        total = hist.sum()
        if total == 0:
            return (np.nan, np.nan)
        com_x = (X * hist).sum() / total
        com_y = (Y * hist).sum() / total
        return (com_x, com_y)

    x_brain = array_mri[brain_only_mask]
    y_brain = array_biasfield[brain_only_mask]
    x_tumor = array_mri[tumor_mask]
    y_tumor = array_biasfield[tumor_mask]

    com_brain = compute_center_of_mass(x_brain, y_brain, bins=bins)
    com_tumor = compute_center_of_mass(x_tumor, y_tumor, bins=bins)
    return com_brain, com_tumor

def compute_all_com_mri_type(new_dir_path, patient_dir_name_nifti, mri_type, patient_number):
    patient_dir_path = os.path.join(new_dir_path, patient_dir_name_nifti)
    array_dir_path = os.path.join(patient_dir_path, 'array')
    if not os.path.isdir(patient_dir_path):
        print(f"Directory {patient_dir_path} does not exist")
        return

    patient_dir_name = patient_dir_name_nifti.split("_")[0]

    # Load native MRI
    native_image_array = np.load(os.path.join(array_dir_path, f'{patient_dir_name}_{mri_type}_rescaled.npy')).astype(np.float32)
    brain_seg_array = np.load(os.path.join(array_dir_path, f'{patient_dir_name}_brain_segmentation_array.npy')).astype(np.float32)
    tumor_binary_array = np.load(os.path.join(array_dir_path, f'{patient_dir_name}_tumor_binary_array.npy')).astype(np.float32)

    # Load all 4 biasfield arrays
    biasfield_names = {
        'N4BB': f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_rescaled.npy',
        'N4HH': f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_rescaled.npy',
        'N4BH': f'biasfield_{patient_dir_name}_{mri_type}_N4_brain_healthy_mask_rescaled.npy',
        'N4HB': f'biasfield_{patient_dir_name}_{mri_type}_N4_healthy_mask_brain_rescaled.npy'
    }

    all_com_patient = {}
    for key, filename in biasfield_names.items():
        path = os.path.join(array_dir_path, filename)
        bias_array = np.load(path).astype(np.float32)
        com_brain, com_tumor = compute_center_of_mass_regions(native_image_array, bias_array, brain_seg_array, tumor_binary_array)
        all_com_patient[f'Patient_{patient_number}_{key}'] = (com_brain, com_tumor)

    return all_com_patient

def process_patient(folder, new_dir, mri_type):
    patient_number = folder.split("-")[2].split("_")[0]
    array_dir = os.path.join(new_dir, folder, "array")
    if not os.path.exists(array_dir):
        print(f"Array directory missing for patient {patient_number}")
        return patient_number, None
    try:
        result = compute_all_com_mri_type(new_dir, folder, mri_type, patient_number)
        return patient_number, result
    except Exception as e:
        print(f"Error processing patient {patient_number}: {e}")
        return patient_number, None

def compute_coms_all_patients(NEW_DIR, mri_type):
    patient_folders = [
        folder for folder in os.listdir(NEW_DIR)
        if os.path.isdir(os.path.join(NEW_DIR, folder)) and folder.startswith(CONTROL1)
    ]

    all_com_bb, all_com_hh, all_com_bh, all_com_hb = {}, {}, {}, {}

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(process_patient, folder) for folder in patient_folders]
        for future in concurrent.futures.as_completed(futures):
            patient_number, all_com_patient = future.result()
            if all_com_patient is None:
                continue
            for correction in all_com_patient:
                if 'N4BB' in correction:
                    all_com_bb[patient_number] = all_com_patient[correction]
                elif 'N4HH' in correction:
                    all_com_hh[patient_number] = all_com_patient[correction]
                elif 'N4BH' in correction:
                    all_com_bh[patient_number] = all_com_patient[correction]
                elif 'N4HB' in correction:
                    all_com_hb[patient_number] = all_com_patient[correction]

    return all_com_bb, all_com_hh, all_com_bh, all_com_hb

# ---------------------- Extract x/y ----------------------
def extract_x_y_all_groups(all_com_bb, all_com_hh, all_com_bh, all_com_hb):
    def extract_xy(com_dict):
        x_brain, y_brain, x_tumor, y_tumor = [], [], [], []
        for patient_id, (com_brain, com_tumor) in com_dict.items():
            if not (np.isnan(com_brain[0]) or np.isnan(com_brain[1])):
                x_brain.append(com_brain[0])
                y_brain.append(com_brain[1])
            if not (np.isnan(com_tumor[0]) or np.isnan(com_tumor[1])):
                x_tumor.append(com_tumor[0])
                y_tumor.append(com_tumor[1])
        return (np.array(x_brain), np.array(y_brain)), (np.array(x_tumor), np.array(y_tumor))

    (x_brain_bb, y_brain_bb), (x_tumor_bb, y_tumor_bb) = extract_xy(all_com_bb)
    (x_brain_hh, y_brain_hh), (x_tumor_hh, y_tumor_hh) = extract_xy(all_com_hh)
    (x_brain_bh, y_brain_bh), (x_tumor_bh, y_tumor_bh) = extract_xy(all_com_bh)
    (x_brain_hb, y_brain_hb), (x_tumor_hb, y_tumor_hb) = extract_xy(all_com_hb)

    results = {
        "brain": {
            "N4BB": (x_brain_bb, y_brain_bb),
            "N4HH": (x_brain_hh, y_brain_hh),
            "N4BH": (x_brain_bh, y_brain_bh),
            "N4HB": (x_brain_hb, y_brain_hb),
        },
        "tumor": {
            "N4BB": (x_tumor_bb, y_tumor_bb),
            "N4HH": (x_tumor_hh, y_tumor_hh),
            "N4BH": (x_tumor_bh, y_tumor_bh),
            "N4HB": (x_tumor_hb, y_tumor_hb),
        },
    }

    return results

# ---------------------- Kruskal–Wallis ----------------------
def kruskal_wallis_com_test(com_data, region="brain", axis="x"):
    if region not in com_data:
        raise ValueError(f"Region '{region}' not in com_data. Must be 'brain' or 'tumor'.")
    if axis not in ["x", "y"]:
        raise ValueError(f"Axis must be 'x' or 'y'.")

    group_data = []
    group_labels = []
    for correction, (x_vals, y_vals) in com_data[region].items():
        arr = x_vals if axis == "x" else y_vals
        if arr.size > 0:
            group_data.append(arr)
            group_labels.append(correction)

    if len(group_data) < 2:
        print(f"Not enough data for Kruskal–Wallis test on {region} {axis}-axis.")
        return None

    # Kruskal–Wallis test
    H, p = stats.kruskal(*group_data)
    print(f"\nKruskal–Wallis test for {region} ({axis}-axis): H={H:.4f}, p={p:.4e}")

    # Dunn post-hoc test
    all_values = np.concatenate(group_data)
    all_groups = np.concatenate([np.full(len(arr), label) for arr, label in zip(group_data, group_labels)])
    df = pd.DataFrame({"value": all_values, "group": all_groups})
    posthoc = sp.posthoc_dunn(df, val_col="value", group_col="group", p_adjust="bonferroni")
    print("\nDunn posthoc test (Bonferroni-corrected p-values):")
    print(posthoc)

    return {"H": H, "p": p, "posthoc": posthoc}

# ---------------------- Plotting ----------------------
def plot_coms_for_all_patients(all_com_bb, all_com_hh, all_com_bh, all_com_hb):
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.ravel()
    corrections = ['N4BB', 'N4HH', 'N4BH', 'N4HB']
    all_com_dicts = [all_com_bb, all_com_hh, all_com_bh, all_com_hb]
    colors = {'brain': 'blue', 'tumor': 'red'}

    for i, (correction, com_dict) in enumerate(zip(corrections, all_com_dicts)):
        ax = axes[i]
        brain_points = [v[0] for v in com_dict.values()]
        tumor_points = [v[1] for v in com_dict.values()]

        brain_points = np.array(brain_points)
        tumor_points = np.array(tumor_points)

        if len(brain_points) == 0 or len(tumor_points) == 0:
            ax.set_title(f'CoM Scatterplot: {correction} (no valid data)')
            continue

        ax.scatter(brain_points[:, 0], brain_points[:, 1], color=colors['brain'], label='Brain-only', alpha=0.6)
        ax.scatter(tumor_points[:, 0], tumor_points[:, 1], color=colors['tumor'], label='Tumor-only', alpha=0.6)

        ax.set_title(f'CoM Scatterplot: {correction}')
        ax.set_xlabel('Native MRI intensity')
        ax.set_ylabel('Biasfield intensity')
        ax.legend()
        ax.grid(True)

    plt.tight_layout()
    plt.show()


# ---------------------- MAIN ----------------------
if __name__ == '__main__':
    import concurrent.futures

    # Ask user which MRI type to analyze
    mri_type = get_user_answer(MRI_TYPE)

    # Collect all patient folders
    patient_folders = [
        folder for folder in os.listdir(NEW_DIR)
        if os.path.isdir(os.path.join(NEW_DIR, folder)) and folder.startswith(CONTROL1)
    ]

    # Run in parallel using top-level function
    all_results = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_patient, folder, NEW_DIR, mri_type)
            for folder in patient_folders
        ]
        for future in concurrent.futures.as_completed(futures):
            all_results.append(future.result())

    # Merge results into separate dictionaries for each correction
    all_com_bb, all_com_hh, all_com_bh, all_com_hb = {}, {}, {}, {}
    for patient_number, all_com_patient in all_results:
        if all_com_patient is None:
            continue
        for correction in all_com_patient:
            if 'N4BB' in correction:
                all_com_bb[patient_number] = all_com_patient[correction]
            elif 'N4HH' in correction:
                all_com_hh[patient_number] = all_com_patient[correction]
            elif 'N4BH' in correction:
                all_com_bh[patient_number] = all_com_patient[correction]
            elif 'N4HB' in correction:
                all_com_hb[patient_number] = all_com_patient[correction]

    # Plot COM scatterplots
    plot_coms_for_all_patients(all_com_bb, all_com_hh, all_com_bh, all_com_hb)

    # Extract x/y values for statistical testing
    com_data = extract_x_y_all_groups(all_com_bb, all_com_hh, all_com_bh, all_com_hb)

    # Run Kruskal–Wallis tests for brain and tumor, x and y axes
    kruskal_wallis_com_test(com_data, region="brain", axis="x")
    kruskal_wallis_com_test(com_data, region="brain", axis="y")
    kruskal_wallis_com_test(com_data, region="tumor", axis="x")
    kruskal_wallis_com_test(com_data, region="tumor", axis="y")
