import os
import subprocess
from pathlib import Path
import pandas as pd
import numpy as np

# ============================================================
#                     CONFIGURATION CONSTANTS
# ============================================================

MAIN_FOLDER = Path("/mnt/external/reorg_patients_UCSF")

OUTPUT_CSV = Path("intensity_report.csv")
SUMMARY_TXT = Path("intensity_summary.txt")

PATIENT_PREFIX = "UCSF-PDGM-"

REG_FOLDER_NAME = "reg"
ANAT_FOLDER_NAME = "anat"

IMAGE_EXTENSION = "*.nii.gz"

INCLUDE_MODALITIES = ["FLAIR", "T1c"]
EXCLUDE_KEYWORDS = ["T2", "T1", "dn", "biasfield"]

OUTLIER_METHOD = "iqr"
ZSCORE_THRESHOLD = 3.0
IQR_MULTIPLIER = 1.5


# ============================================================
#                     FSL UTILS
# ============================================================

def fslstats(image_path: Path):
    """Return min, max, mean, std, p95."""
    try:
        cmd = ["fslstats", str(image_path), "-R", "-m", "-s", "-P", "95"]
        result = subprocess.check_output(cmd, text=True).strip().split()
        min_val, max_val, mean_val, std_val, p95 = map(float, result)
        return min_val, max_val, mean_val, std_val, p95
    except Exception as e:
        print(f"⚠ Error reading {image_path}: {e}")
        return None, None, None, None, None


# ============================================================
#                    OUTLIER DETECTION
# ============================================================

def detect_outliers(series: pd.Series, method="iqr"):
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
#                     FILTERING LOGIC
# ============================================================

def should_process_image(image_name: str):
    if not any(mod in image_name for mod in INCLUDE_MODALITIES):
        return False
    if any(excl in image_name for excl in EXCLUDE_KEYWORDS):
        return False
    return True


# ============================================================
#                     MAIN EXTRACTION
# ============================================================

def collect_intensity_stats():
    data = []

    for patient_dir in os.listdir(MAIN_FOLDER):
        patient_number = patient_dir.split("_")[1]
        patient_number -= (PATIENT_PREFIX)
        print(f"\n📂 Processing {patient_dir}")

        for folder in [f"{patient_dir}/{REG_FOLDER_NAME}", f"{patient_dir}/{ANAT_FOLDER_NAME}"]:

            for image_path_name in folder:

                if not should_process_image(image_path_name):
                    continue

                # -----------------------------------
                # SAFE PARSING: ensure modality + variant exist
                # -----------------------------------
                parts = image_path_name.split("_")

                modality = parts[0]
                variant = parts[-1]
                label = f"{modality}_{variant}"

                image_path = os.path(f"{MAIN_FOLDER}/{patient_dir}/{folder}/{image_path_name}")

                minv, maxv, meanv, stdv, p95 = fslstats(image_path)

                row = {
                    "patient": patient_dir.name,
                    "image": image_path.name,
                    f"min_{label}": minv,
                    f"max_{label}": maxv,
                    f"mean_{label}": meanv,
                    f"std_{label}": stdv,
                    f"p95_{label}": p95
                }

                data.append(row)

    # Build DataFrame
    df = pd.DataFrame(data)

    # -----------------------------------
    # SAFETY CHECKS
    # -----------------------------------
    if df.empty:
        raise RuntimeError("❌ No valid images found. DataFrame is empty.")

    if "patient" not in df.columns:
        print(df.head())
        raise RuntimeError("❌ 'patient' column missing — filenames likely malformed.")

    value_columns = [c for c in df.columns if c not in ("patient", "image")]

    df = df.pivot_table(
        index="patient",
        values=value_columns,
        aggfunc="first"
    ).reset_index()

    return df


# ============================================================
#                     SUMMARY REPORT
# ============================================================

def write_summary(df: pd.DataFrame, output_file: Path):

    min_cols = [c for c in df.columns if c.startswith("min_")]
    max_cols = [c for c in df.columns if c.startswith("max_")]
    mean_cols = [c for c in df.columns if c.startswith("mean_")]
    std_cols = [c for c in df.columns if c.startswith("std_")]
    p95_cols = [c for c in df.columns if c.startswith("p95_")]

    all_cols = min_cols + max_cols + mean_cols + std_cols + p95_cols

    with open(output_file, "w") as f:

        f.write("==== INTENSITY SUMMARY REPORT ====\n\n")
        f.write(df[all_cols].describe().to_string())
        f.write("\n\nOutlier method: " + OUTLIER_METHOD + "\n\n")

        groups = {
            "min": min_cols,
            "max": max_cols,
            "mean": mean_cols,
            "p95": p95_cols,
        }

        for name, cols in groups.items():
            f.write(f"\n--- OUTLIERS for {name} ---\n")
            for col in cols:
                outliers = detect_outliers(df[col])
                if outliers.any():
                    f.write(f"\nColumn: {col}\n")
                    f.write(df.loc[outliers, ["patient", col]].to_string() + "\n")
                else:
                    f.write(f"\nColumn {col}: No outliers.\n")

        f.write("\nReport completed.\n")


# ============================================================
#                     MAIN
# ============================================================

if __name__ == "__main__":
    df = collect_intensity_stats()
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ Saved: {OUTPUT_CSV.resolve()}")

    write_summary(df, SUMMARY_TXT)
    print(f"📄 Summary saved: {SUMMARY_TXT.resolve()}")
