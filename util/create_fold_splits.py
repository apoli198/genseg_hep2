import os
import pandas as pd
from sklearn.model_selection import StratifiedKFold

# Paths
images_folder = './data/HEp-2_specimen/train_patch'
input_csv = './data/HEp-2_specimen/train.csv'
output_csv = './data/HEp-2_specimen/train_folds.csv'
n_folds = 5

# Read the train.csv to get the intensity information
train_df = pd.read_csv(input_csv)

# Extract patient IDs and intensities
train_df['patient_id'] = train_df['ID'].astype(str).str.zfill(5)

# Mapping of patient_id to intensity
patient_intensity_map = dict(zip(train_df['patient_id'], train_df['intensity']))

# List all files in the image folder
image_files = os.listdir(images_folder)

# Collect image filename + intensity + patient_id
image_info = []
for filename in image_files:
    if filename.endswith('.tif') and '_Mask' not in filename:
        patient_id = filename.split('_')[0]  # Extract patient_id (e.g., 00001)
        intensity = patient_intensity_map.get(patient_id)
        image_info.append({'filename': filename, 'intensity': intensity, 'patient_id': patient_id})

# Build DataFrame
image_df = pd.DataFrame(image_info)

# Sort by patient_id
image_df = image_df.sort_values(by='patient_id')

# === Correct splitting ===
# Unique patients with intensity
patients = image_df[['patient_id', 'intensity']].drop_duplicates()

# Initialize StratifiedKFold
skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
patients['fold'] = -1  # Initialize fold column

# Stratified splitting by patient
for fold, (train_idx, val_idx) in enumerate(skf.split(patients, patients['intensity'])):
    patients.loc[patients.index[val_idx], 'fold'] = fold

# Merge fold assignment back to the image-level DataFrame
image_df = image_df.merge(patients[['patient_id', 'fold']], on='patient_id', suffixes=('', '_patient'))

# === Remove '.tif' extension from filename ===
image_df['filename'] = image_df['filename'].str.replace('.tif', '', regex=False)

# Output CSV format
output_df = image_df[['filename', 'intensity', 'fold', 'patient_id']]

# Extract the polarization number as an integer
output_df['polarization_number'] = output_df['filename'].str.split('_').str[1].str.replace('p', '').astype(int)

# Extract the patch number as an integer
output_df['patch_number'] = output_df['filename'].str.split('patch').str[1].astype(int)

# Sort the DataFrame by 'patient_id', 'polarization_number', and then 'patch_number'
output_df = output_df.sort_values(by=['patient_id', 'polarization_number', 'patch_number']).reset_index(drop=True)

# Drop the temporary columns
output_df = output_df.drop(columns=['polarization_number', 'patch_number'])

# Save the CSV
output_df.to_csv(output_csv, index=False)

print('CSV with fold information created:', output_csv)