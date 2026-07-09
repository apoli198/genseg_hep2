import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import multiprocessing as mp
from collections import defaultdict
from typing import List, Dict, Optional, Tuple

def analyze_problematic_masks(data_dir: str, low_coverage_threshold: float = 0.5, high_coverage_threshold: float = 80.0) -> Dict[str, List[Dict]]:
    """
    Analyzes all mask files to identify masks that are all black, have low coverage, or have high coverage.
    Returns a dictionary of problematic patches grouped by their complete mask ID.
    
    Args:
        data_dir (str): The path to the data directory containing the 'train_patch' folder.
        low_coverage_threshold (float): The percentage threshold below which a mask is considered low coverage.
        high_coverage_threshold (float): The percentage threshold above which a mask is considered high coverage.

    Returns:
        Dict[str, List[Dict]]: A dictionary where keys are mask_ids and values are
                               lists of dictionaries with information about each problematic patch.
    """
    mask_folder = os.path.join(data_dir, "train_patch")
    problematic_patches_by_mask = defaultdict(list)
    
    try:
        all_mask_files = [f for f in os.listdir(mask_folder) if f.endswith("_Mask.tif")]
    except FileNotFoundError:
        print(f"Error: 'train_patch' folder not found in {data_dir}.")
        return {}

    print(f"Analyzing all {len(all_mask_files)} mask files.")
    
    for mask_file in all_mask_files:
        mask_file_path = os.path.join(mask_folder, mask_file)
        filename = mask_file.replace('_Mask.tif', '')
        
        # Extract the complete mask ID (e.g., '00002_p0')
        parts = filename.split('_')
        if len(parts) >= 2:
            mask_id = f"{parts[0]}_{parts[1]}"
        else:
            continue
        
        try:
            mask_image = Image.open(mask_file_path).convert('L')
            mask_array = np.asarray(mask_image)
            
            non_zero_count = np.count_nonzero(mask_array)
            mask_size = mask_array.size
            
            if mask_size == 0:
                continue

            mask_coverage_percentage = (non_zero_count / mask_size) * 100
            
            if non_zero_count == 0 or np.max(mask_array) <= 1:
                problematic_patches_by_mask[mask_id].append({
                    'filename': filename,
                    'issue': 'All_black_mask',
                    'details': "Mask contains only zero or very low pixel values."
                })
            
            elif mask_coverage_percentage < low_coverage_threshold:
                problematic_patches_by_mask[mask_id].append({
                    'filename': filename,
                    'issue': 'Low_coverage_mask',
                    'details': f"Non-zero pixel percentage: {mask_coverage_percentage:.2f}%"
                })
            
            elif mask_coverage_percentage > high_coverage_threshold:
                problematic_patches_by_mask[mask_id].append({
                    'filename': filename,
                    'issue': 'High_coverage_mask',
                    'details': f"Non-zero pixel percentage: {mask_coverage_percentage:.2f}%"
                })
        
        except FileNotFoundError:
            print(f"File not found: {mask_file_path}. Skipping.")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    return problematic_patches_by_mask

def visualize_complete_mask_grid(mask_id: str, filenames: List[str], data_dir: str, output_dir: str):
    """
    Creates a single 5x6 grid visualization for a complete mask, showing all its patches and masks.
    """
    grid_rows = 5
    grid_cols = 6
    fig, axes = plt.subplots(grid_rows * 2, grid_cols, figsize=(25, 20))
    output_path = os.path.join(output_dir, f'mask_{mask_id}_grid.png')
    filenames.sort(key=lambda f: int(f.split('_patch')[1]))

    for i, filename in enumerate(filenames):
        patch_row = i // grid_cols
        patch_col = i % grid_cols
        img_file_path = os.path.join(data_dir, "train_patch", f"{filename}.tif")
        mask_file_path = os.path.join(data_dir, "train_patch", f"{filename}_Mask.tif")
        
        try:
            original_image = Image.open(img_file_path).convert('L')
            mask_image = Image.open(mask_file_path).convert('L')
            
            img_ax = axes[patch_row * 2, patch_col]
            img_ax.imshow(np.asarray(original_image), cmap='gray')
            img_ax.set_title(f'Image: {filename}', fontsize=8)
            img_ax.axis('off')

            mask_ax = axes[patch_row * 2 + 1, patch_col]
            mask_ax.imshow(np.asarray(mask_image), cmap='gray')
            mask_ax.set_title(f'Mask: {filename}_Mask', fontsize=8)
            mask_ax.axis('off')
        
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    for row in range(i * 2 + 2, grid_rows * 2):
        for col in range(grid_cols):
            axes[row, col].axis('off')
            
    plt.tight_layout()
    plt.savefig(output_path, dpi=100)
    plt.close(fig)
    print(f"Visualization grid for mask {mask_id} saved to {output_path}")

def generate_visualizations_for_faulty_masks(data_dir: str, output_dir: str):
    """
    Orchestrates the analysis and visualization process for faulty masks.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    print("=== STEP 1: Identifying problematic masks ===")
    problematic_patches_by_mask = analyze_problematic_masks(data_dir)

    print("\n=== ANALYSIS RESULTS ===")
    if not problematic_patches_by_mask:
        print("No masks were identified with low coverage, high coverage, or as all black.")
        return
        
    total_problematic_patches = sum(len(v) for v in problematic_patches_by_mask.values())
    print(f"Found {total_problematic_patches} problematic patches across {len(problematic_patches_by_mask)} complete masks.")
    
    mask_folder = os.path.join(data_dir, "train_patch")
    all_patch_filenames = [f.replace('_Mask.tif', '') for f in os.listdir(mask_folder) if f.endswith("_Mask.tif")]

    all_masks_groups = defaultdict(list)
    for filename in all_patch_filenames:
        parts = filename.split('_')
        if len(parts) >= 2:
            mask_id = f"{parts[0]}_{parts[1]}"
            all_masks_groups[mask_id].append(filename)

    print("\n=== STEP 2: Generating visualizations for all patches in faulty masks ===")
    problematic_mask_ids = list(problematic_patches_by_mask.keys())
    print(f"Visualizing {len(problematic_mask_ids)} complete masks with faulty patches.")
    
    for mask_id in problematic_mask_ids:
        filenames_for_mask = all_masks_groups.get(mask_id, [])
        if filenames_for_mask:
            visualize_complete_mask_grid(mask_id, filenames_for_mask, data_dir, output_dir)
        else:
            print(f"Warning: No patches found for mask ID {mask_id}. Skipping visualization.")
    
    print(f"\nVisualizations for all faulty masks have been saved to the '{output_dir}' directory.")

if __name__ == "__main__":
    data_directory = "/user/apoli/GenSeg/data/HEp-2_specimen"
    output_directory = "./visualizations_faulty_masks_full"
    
    generate_visualizations_for_faulty_masks(data_directory, output_directory)