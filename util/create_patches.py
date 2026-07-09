# Fixed patch creation script - processes image and mask pairs together
import os
from PIL import Image

IMAGE_WIDTH = 1388
IMAGE_HEIGHT = 1040
PATCH_SIZE = 256
X_COORDS = [0, 226, 453, 679, 906, 1132]
Y_COORDS = [0, 196, 392, 588, 784]

images_folder = './data/HEp-2_specimen/train'
patches_folder = './data/HEp-2_specimen/train_patch'

print("Creating patches...")

# Get all non-mask image files
image_files = []
for filename in os.listdir(images_folder):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp')) and '_Mask' not in filename:
        image_files.append(filename)

print(f"Found {len(image_files)} image files to process")

# Process each image-mask pair together
for filename in sorted(image_files):  # Sort for consistent processing
    base_name = os.path.splitext(filename)[0]
    
    # Construct paths
    image_path = os.path.join(images_folder, filename)
    mask_filename = f"{base_name}_Mask.tif"  # Assume mask has .tif extension
    mask_path = os.path.join(images_folder, mask_filename)
    
    # Check if both files exist
    if not os.path.exists(image_path):
        print(f"Warning: Image file not found: {image_path}")
        continue
    if not os.path.exists(mask_path):
        print(f"Warning: Mask file not found: {mask_path}")
        continue
    
    try:
        # Load both image and mask
        image = Image.open(image_path)
        mask = Image.open(mask_path)
        
        # Verify dimensions match
        if image.size != mask.size:
            print(f"Warning: Size mismatch for {base_name}: image {image.size}, mask {mask.size}")
            continue
        
        print(f"Processing {base_name}...")
        
        # Create patches for both image and mask simultaneously
        patch_counter = 0
        for j in Y_COORDS:
            for i in X_COORDS:
                box = (i, j, i + PATCH_SIZE, j + PATCH_SIZE)
                
                # Create image patch
                image_patch = image.crop(box)
                image_patch_filename = f"{base_name}_patch{patch_counter}.tif"
                image_patch.save(os.path.join(patches_folder, image_patch_filename))
                
                # Create mask patch
                mask_patch = mask.crop(box)
                mask_patch_filename = f"{base_name}_patch{patch_counter}_Mask.tif"
                mask_patch.save(os.path.join(patches_folder, mask_patch_filename))
                
                patch_counter += 1
        
        print(f"  Created {patch_counter} patch pairs for {base_name}")
        
    except Exception as e:
        print(f"Error processing {base_name}: {e}")
        continue

print("Patches created successfully.")

# Verification step
print("\n=== VERIFICATION ===")
image_patches = []
mask_patches = []

for filename in os.listdir(patches_folder):
    if filename.endswith('.tif'):
        if '_Mask.tif' in filename:
            mask_patches.append(filename)
        else:
            image_patches.append(filename)

image_patches.sort()
mask_patches.sort()

print(f"Created {len(image_patches)} image patches")
print(f"Created {len(mask_patches)} mask patches")

# Check for missing pairs
missing_pairs = []
for img_patch in image_patches:
    expected_mask = img_patch.replace('.tif', '_Mask.tif')
    if expected_mask not in mask_patches:
        missing_pairs.append(img_patch)

if missing_pairs:
    print(f"WARNING: {len(missing_pairs)} image patches missing corresponding masks:")
    for missing in missing_pairs[:10]:  # Show first 10
        print(f"  {missing}")
else:
    print("All image patches have corresponding mask patches")