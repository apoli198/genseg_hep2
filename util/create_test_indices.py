import os
import pandas as pd
import re

images_folder = './data/HEp-2_specimen/test_patch'
input_csv    = './data/HEp-2_specimen/test.csv'
output_csv   = './data/HEp-2_specimen/test_indices.csv'

# Read the test.csv to get the intensity information
test_df = pd.read_csv(input_csv)
# Convert 'ID' to a zero-padded string to match filenames
test_df['patient_id'] = test_df['ID'].astype(str).str.zfill(5)
# Create a dictionary to map patient IDs to intensity values
patient_intensity_map = dict(zip(test_df['patient_id'], test_df['intensity']))

# Helper to extract the numerical part of the polarization
def get_polarization_idx(fn):
    m = re.search(r'_p(\d+)', fn)
    return int(m.group(1)) if m else -1

# Helper to extract the numerical part of the patch index
def get_patch_idx(fn):
    m = re.search(r'_patch(\d+)', fn)
    return int(m.group(1)) if m else -1

image_info = []
# Iterate through all files in the images_folder
for fn in os.listdir(images_folder):
    # Process only .tif files that are not masks
    if not fn.endswith('.tif') or '_Mask' in fn:
        continue
    # Extract the patient ID from the filename
    pid = fn.split('_')[0]
    # Append a dictionary with the extracted information to the list
    image_info.append({
        'filename':     fn,
        'intensity':    patient_intensity_map.get(pid),
        'patient_id':   pid,
        'polarization_idx': get_polarization_idx(fn),
        'patch_idx':    get_patch_idx(fn)
    })

# Create a DataFrame from the list of dictionaries
df = pd.DataFrame(image_info)

# Sort the DataFrame first by patient ID, then by the numerical polarization index, and finally by the numerical patch index
df = df.sort_values(['patient_id', 'polarization_idx', 'patch_idx'])

# Drop the helper columns as they are no longer needed for the final output
df = df.drop(columns=['polarization_idx', 'patch_idx'])

# Save the sorted DataFrame to a new CSV file
df.to_csv(output_csv, index=False)
print(f"CSV file '{output_csv}' created with {len(df)} entries.")