import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
#                     CONFIGURATION CONSTANTS
# ============================================================

# Root directory containing patient folders
MAIN_FOLDER = Path("mnt/external/reorg_patients_UCSF")

# Output CSV and summary report
OUTPUT_CSV = Path("mnt/external/intensity_report.csv")
SUMMARY_TXT = Path("mnt/external/intensity_summary.txt")

# Patient folder naming
PATIENT_PREFIX = "UCSF-PDGM-"            # Pattern for patient folders

# Subfolder names
REG_FOLDER_NAME = "reg"                # Folder containing .nii.gz images
ANAT_FOLDER_NAME = "anat"              # (Optional – included if you need it)

# Image extension to search
IMAGE_EXTENSION = "*.nii.gz"

# Outlier detection method ("zscore" or "iqr")
OUTLIER_METHOD = "iqr"

# Z-score threshold for outliers (if method="zscore")
ZSCORE_THRESHOLD = 3.0

# IQR multiplier for outliers (if method="iqr")
IQR_MULTIPLIER = 1.5

# ============================================================
#                     FSL UTILS
# ============================================================

def fslstats(image_path: Path):
    """Return min, max, mean, std, p95 for an image using fslstats."""
    try:
        # -R = min max
        # -m = mean
        # -s = std
        # -P 95 = 95th percentile
        cmd = [
            "fslstats", str(image_path),
            "-R", "-m", "-s", "-P", "95"
        ]
        result = subprocess.check_output(cmd, text=True).strip().split()
        min_val, max_val, mean_val, std_val, p95 = map(float, result)
        return min_val, max_val, mean_val, std_val, p95

    except Exception as e:
        print(f"⚠ Error reading {image_path}: {e}")
        return None, None, None, None, None


# ============================================================
#                     OUTLIER DETECTION
# ============================================================

def detect_outliers(series: pd.Series, method="iqr"):
    """Return boolean mask indicating outliers."""
    if method == "zscore":
        zscores = np.abs((series - series.mean()) / series.std())
        return zscores > ZSCORE_THRESHOLD

    elif method == "iqr":
        q1, q3 = series.quantile([0.25, 0.75])
        iqr = q3 - q1
        lower = q1 - IQR_MULTIPLIER * iqr
        upper = q3 + IQR_MULTIPLIER * iqr
        return (series < lower) | (series > upper)

    else:
        raise ValueError("Unknown outlier detection method.")


# ============================================================
#                     MAIN EXTRACTION LOOP
# ============================================================

def collect_intensity_stats():
    data = []

    for patient_dir in sorted(MAIN_FOLDER.glob(f"{PATIENT_PREFIX}*")):
        print(f"\n📂 Processing {patient_dir.name}")

        # Search for nii.gz in selected subfolders
        candidate_dirs = [
            patient_dir / REG_FOLDER_NAME,
            patient_dir / ANAT_FOLDER_NAME,
        ]

        for folder in candidate_dirs:
            if not folder.exists():
                continue

            for image_path in folder.glob(IMAGE_EXTENSION):
                if "dn" and "biasfield" and "T2" in image_path.name:
                    continue
                minv, maxv, meanv, stdv, p95 = fslstats(image_path)
                data.append({
                    "patient": patient_dir.name,
                    "folder": folder.name,
                    "image": image_path.name,
                    "min": minv,
                    "max": maxv,
                    "mean": meanv,
                    "std": stdv,
                    "p95": p95
                })

    df = pd.DataFrame(data)
    return df


# ============================================================
#                     SUMMARY REPORT
# ============================================================

def write_summary(df: pd.DataFrame, output_file: Path):
    with open(output_file, "w") as f:
        f.write("==== INTENSITY SUMMARY REPORT ====\n\n")
        f.write("Global intensity statistics across all images:\n")
        f.write(df[["min", "max", "mean", "std", "p95"]].describe().to_string())
        f.write("\n\n")

        f.write("Outlier detection method: " + OUTLIER_METHOD + "\n\n")

        for col in ["min", "max", "mean", "p95"]:
            f.write(f"--- OUTLIERS in {col} ---\n")
            outliers = detect_outliers(df[col], OUTLIER_METHOD)
            if outliers.any():
                f.write(df[outliers][["patient", "image", col]].to_string())
            else:
                f.write("No outliers detected.\n")
            f.write("\n")

        f.write("\nReport completed.\n")


# ============================================================
#                     MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    df = collect_intensity_stats()
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n✅ Intensity statistics saved to: {OUTPUT_CSV.resolve()}")

    write_summary(df, SUMMARY_TXT)
    print(f"📄 Summary report written to: {SUMMARY_TXT.resolve()}")
