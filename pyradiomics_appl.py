from pathlib import Path
import pandas as pd
from radiomics import featureextractor

# ============================================================
#                     CONFIGURATION CONSTANTS
# ============================================================

# --- Paths ---
MAIN_FOLDER = Path("/mnt/external/reorg_patients_UCSF")          # Root directory containing patient folders
OUTPUT_CSV = Path("radiomics_features.csv")         # Output CSV file path

# --- MRI Modalities and N4 Variants ---
MODALITIES = ["FLAIR", "T1c"]                       # Modalities to process

# native, n4bb, n4hb
IMAGES_TO_INCLUDE = ["brain.nii.gz", "healthy_mask_brain.nii.gz"]      # Correction variants to include

# --- File Naming ---
SEGMENTATION_FILENAME = "segmentation.nii.gz"       # File name in seg/ folder
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
PATIENT_PREFIX = "patient_"                         # Pattern prefix for patient folders
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
        print(f"⚠️ Error processing {image_path.name}: {e}")
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

    for patient_dir in sorted(main_folder.glob(f"{PATIENT_PREFIX}*")):
        print(f"\n📂 Processing {patient_dir.name}")

        seg_path = patient_dir / SEG_FOLDER_NAME / SEGMENTATION_FILENAME
        if not seg_path.exists():
            print(f"⚠️ Missing segmentation in {patient_dir.name}, skipping.")
            continue

        reg_dir = patient_dir / REG_FOLDER_NAME
        if not reg_dir.exists():
            print(f"⚠️ Missing reg/ folder in {patient_dir.name}, skipping.")
            continue

        patient_features = {ID_COLUMN: patient_dir.name}

        # Iterate through modalities and correction variants
        for modality in MODALITIES:
            for variant in N4_VARIANTS:
                img_path = reg_dir / f"{modality}_{variant}.nii.gz"
                if not img_path.exists():
                    print(f"   ⚠️ Missing {img_path.name}, skipping this combination.")
                    continue

                feats = extract_features(img_path, seg_path)
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
