from pathlib import Path
import pandas as pd
import os
from radiomics import featureextractor

# ============================================================
#                     CONFIGURATION CONSTANTS
# ============================================================

# --- Paths ---
MAIN_FOLDER = Path("/mnt/external/reorg_patients_UCSF")          # Root directory containing patient folders
OUTPUT_CSV = Path("/mnt/external/radiomics_features.csv")         # Output CSV file path

# --- MRI Modalities and N4 Variants ---
MODALITIES = ["FLAIR", "T1c"]                       # Modalities to process

# native, n4bb, n4hb
N4_VARIANTS = ["brain", "healthy_mask_brain"]      # Correction variants to include

# --- File Naming ---
SEGMENTATION_FILENAME = "segmentation.nii.gz"       # File name in seg/ folder
ANAT_FOLDER_NAME = "anat"                           # Subfolder containing the native images
REG_FOLDER_NAME = "reg"                             # Subfolder containing the registered images
SEG_FOLDER_NAME = "seg"                             # Subfolder containing the segmentation mask

# --- PyRadiomics Parameters ---
# DA CONTROLLARE TUTTI
PYRADIOMICS_PARAMS = {
    "binWidth": 25,
    "resampledPixelSpacing": None,
    "interpolator": "sitkBSpline", # CONTROLLA
    "enableCExtensions": True,
}
# CONTROLLA SE IMMAGINI CORRETTE ARRIVANO A 65000
# CONTROLLA SE CI SONO IMMAGINI CON RANGE DI INTENSITà -> CONTROLLA FSLSTATS
# OPPURE CONTROLLA NEL NIFTI SE LE INTENSITà SONO INTERE E IN QUANTI BIT SONO SALVATI

# --- Output Control ---
PATIENT_PREFIX = "UCSF-PDGM-"                         # Pattern prefix for patient folders
ID_COLUMN = "patient_id"                            # Name of ID column in CSV

# ============================================================
#                     FEATURE EXTRACTION
# ============================================================

# Initialize the feature extractor
extractor = featureextractor.RadiomicsFeatureExtractor(**PYRADIOMICS_PARAMS)
extractor.enableAllFeatures()  # Enable all feature classes (customizable)

print("✅ PyRadiomics feature classes enabled:")
for feature_class, features in extractor.enabledFeatures.items():
    print(f"  - {feature_class}: {features}")

# ------------------------------------------------------------
# Helper Functions
# ------------------------------------------------------------
def extract_features(image_path: Path, mask_path: Path) -> dict:
    """Extract PyRadiomics features for a given image–mask pair."""
    try:
        result = extractor.execute(str(image_path), str(mask_path))
        # Remove diagnostic keys (metadata)
        return {k: v for k, v in result.items() if not k.startswith("diagnostics")}
    except Exception as e:
        print(f"Error processing {image_path.name}: {e}")
        return {}

def rename_features(features: dict, modality: str, variant: str) -> dict:
    """Append modality and correction variant suffix to each feature name."""
    return {f"{feat}_{modality}_{variant}": val for feat, val in features.items()}

# ------------------------------------------------------------
# Main Loop
# ------------------------------------------------------------
def process_all_patients(main_folder: Path) -> pd.DataFrame:
    """Iterate through all patients and collect radiomics features."""
    all_data = []

    for patient_dir in os.listdir(MAIN_FOLDER):
        patient_number = patient_dir.split("_")[0]
        print(f"\n📂 Processing {patient_number}")

        anat_path = os.path.join(MAIN_FOLDER, patient_dir, ANAT_FOLDER_NAME)
        if not os.path.isdir(anat_path):
            print(f"Missing anat in {patient_number}, skipping.")
            continue

        seg_path = os.path.join(MAIN_FOLDER, patient_dir, SEG_FOLDER_NAME)
        if not os.path.isdir(seg_path):
            print(f"Missing seg in {patient_number}, skipping.")
            continue
        print("Find the tumor binary mask")

        reg_path = os.path.join(MAIN_FOLDER, patient_dir, REG_FOLDER_NAME)
        if not os.path.isdir(reg_path):
            print(f"Missing reg in {patient_number}, skipping.")
            continue

        patient_features = {ID_COLUMN: patient_number}
        tumor_mask_binary_path = os.path.join(seg_path, "tumor_binary.nii.gz")
        if not os.path.isfile(tumor_mask_binary_path):
            print("Missing segmentation mask, skipping")
            continue
        # Native images
        for modality in MODALITIES:
            variant = "nat"
            img_path_native = os.path.join(anat_path, f"{patient_number}_{modality}.nii.gz")
            if not os.path.isfile(img_path_native):
                print(f"Missing {modality} for {patient_number}, skipping.")
                continue
            feats = extract_features(img_path_native, tumor_mask_binary_path)
            renamed_feats = rename_features(feats, modality, variant)
            patient_features.update(renamed_feats)

        # Iterate through modalities and correction variants
        for modality in MODALITIES:
            img_path_native = os.path.join(anat_path, f"{patient_number}_{modality}.nii.gz")
            if not os.path.isfile(img_path_native):
                print(f"Missing {modality} for {patient_number}, skipping.")
                continue
            for variant in N4_VARIANTS:
                # N4 images
                img_path = os.path.join(reg_path, f"{patient_number}_{modality}_N4_{variant}.nii.gz")
                if not os.path.isfile(img_path):
                    print(f"Missing {modality} and {variant} for {patient_number}, skipping.")
                    continue

                feats = extract_features(img_path, tumor_mask_binary_path)
                renamed_feats = rename_features(feats, modality, variant)
                patient_features.update(renamed_feats)

        all_data.append(patient_features)

    df = pd.DataFrame(all_data)
    if ID_COLUMN in df.columns:
        df.set_index(ID_COLUMN, inplace=True)
    return df

# ------------------------------------------------------------
# Run and Save
# ------------------------------------------------------------
if __name__ == "__main__":
    df = process_all_patients(MAIN_FOLDER)
    df.sort_index(inplace=True)
    df.to_csv(OUTPUT_CSV)
    print(f"\n✅ Radiomics feature extraction complete.")
    print(f"📄 Results saved to: {OUTPUT_CSV.resolve()}")
