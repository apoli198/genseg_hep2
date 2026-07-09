import gc
import os
import sys
sys.path.append('.')
import traceback
import argparse
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging
import cv2

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

from util.HEp2_loader import TestDataset

from unet import UNet

def dice_coeff(pred: Tensor, target: Tensor, epsilon: float = 1e-8):
    """
    Compute the "strict" or "hard" Dice coefficient for testing/evaluation.
    Args:
        pred: Predicted binary mask (0 or 1 values)
        target: Ground truth binary mask (0 or 1 values)
        epsilon: Small value to avoid division by zero
    
    Returns:
        float: Dice coefficient
    """
    pred = (pred > 0.5).float()
    target = (target > 0.5).float()

    volume_sum = pred.sum() + target.sum() + epsilon
    volume_intersect = (pred * target).sum()
    
    return 2 * volume_intersect / volume_sum

def patch_reconstruction(mask_patches, device):
    """
    Reconstructs a full segmentation mask from patches created with the
    specific hardcoded coordinates.

    Args:
        mask_patches: Tensor of mask patches, Shape should be (30, 1, 256, 256).

    Returns:
        torch.Tensor: The reconstructed full segmentation mask.
    """
    # --- Hardcoded parameters from the creation script ---
    IMAGE_WIDTH = 1388
    IMAGE_HEIGHT = 1040
    PATCH_SIZE = 256
    X_COORDS = torch.tensor([0, 226, 453, 679, 906, 1132], device=device)
    Y_COORDS = torch.tensor([0, 196, 392, 588, 784], device=device)

    # --- Reconstruction Logic ---
    # Create a blank canvas using the data type of the input patches
    reconstructed_mask = torch.zeros((IMAGE_HEIGHT, IMAGE_WIDTH), dtype=mask_patches.dtype, device=device)

    patch_index = 0
    # Loop through the coordinates in the exact same order as patch creation
    for j in Y_COORDS:
        for i in X_COORDS:
            # Place each patch onto the canvas at its specific (x, y) location.
            # This "winner takes all" approach simply overwrites the class labels
            # in the overlapping regions, which is a valid way to stitch masks.
            reconstructed_mask[j : j + PATCH_SIZE, i : i + PATCH_SIZE] = mask_patches[patch_index].squeeze(0)
            patch_index += 1

    return reconstructed_mask

def process_batch(batch, net, device):
    """Process a single batch of data"""
    patches = batch['patches'][0].to(device=device, dtype=torch.float32)
    whole_mask_true = batch['whole_mask'][0].to(device=device, dtype=torch.float32)

    with autocast(enabled=True):
        mask_pred = torch.sigmoid(net(patches))
    
    return mask_pred, whole_mask_true

def save_visualizations(masks, whole_mask_pred, whole_mask_true, fold_save_dir, batch_idx):
    """Save visualizations of patches and reconstructed masks"""
    # Save patch visualizations
    _, axs = plt.subplots(5, 6, figsize=(15, 12))
    for j in range(30):
        row = j // 6        # 5 rows
        col = j % 6         # 6 columns
        axs[row, col].imshow(masks[j].squeeze(), cmap='gray')
        axs[row, col].set_title(f"Patch {j}")
        axs[row, col].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_save_dir, f"masks_{batch_idx}.png"))
    plt.close()

    # Save reconstructed mask comparison
    _, axs = plt.subplots(1, 2, figsize=(15, 12))
    axs[0].imshow(whole_mask_true, cmap='gray')
    axs[0].set_title("Ground Truth Mask")
    axs[0].axis('off')
    axs[1].imshow(whole_mask_pred, cmap='gray')
    axs[1].set_title("Predicted Mask")
    axs[1].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join(fold_save_dir, f"reconstructed_mask_{batch_idx}.png"))
    plt.close()

def main ():
    # ---------------------------
    # Parse command-line arguments
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--opt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.opt_path, "rb") as f:
        opt = pickle.load(f)

    # ---------------------------
    # Set up device and directories
    # ---------------------------
    assert opt.cuda_index == int(opt.gpu_ids[0]), "GPU indices should be the same"
    device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')
    data_dir = opt.dataroot
    fold_save_dir = args.save_path + f"-fold{args.fold}"
    os.makedirs(fold_save_dir, exist_ok=True)
    model_dir = os.path.join(opt.model_dir, "end2end-HEp2-unet")
    fold_model_dir = model_dir + f"-fold{args.fold}" + "/final.pkl"
    metrics_file = fold_save_dir + "/metrics.pkl"

    # ---------------------------
    # Set up logging
    # ---------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)

    logger.info(f"starting test fold {args.fold}")

    # ---------------------------
    # Load model and dataset
    # ---------------------------
    net = UNet(n_channels=opt.output_nc, n_classes=opt.output_nc, bilinear=True)
    if not os.path.exists(fold_model_dir):
        logger.error(f"Model file {fold_model_dir} does not exist. Please check the path.")
        exit(1)
    net.load_state_dict(torch.load(fold_model_dir, map_location=device), strict=False)
    net.to(device=device)
    net.eval()

    logger.info("Loading dataset")
    test_set = TestDataset(data_dir, 1.0)
    logger.info(f"Test set size: {len(test_set)}")

    try:
        loader_args = dict(batch_size=opt.batch_size, num_workers=8, pin_memory=True)
        test_loader = DataLoader(test_set, shuffle=False, **loader_args)
        logger.info(f"Test DataLoader created with {len(test_loader)} batches")

        # ---------------------------
        # Evaluate the model
        # ---------------------------
        logger.info("Starting evaluation")
        dice_score_total = 0
        segmentation_accuracy_total = 0
        total_images = len(test_loader)
        viz_interval = max(1, total_images // 4)

        # iterate over the test set
        with torch.no_grad():
            for i, batch in enumerate(test_loader):
                masks_pred, whole_mask_true = process_batch(batch, net, device)

                # Reconstruct the full mask from patches
                whole_mask_pred = patch_reconstruction(masks_pred, device)

                # Convert to numpy for morphological operations
                mask_np = whole_mask_pred.cpu().numpy()
                mask_np = (mask_np > 0.5).astype(np.uint8)
                kernel = np.ones((5, 5), np.uint8)
                mask_np = cv2.morphologyEx(mask_np, cv2.MORPH_CLOSE, kernel)
                # Convert back to tensor
                whole_mask_pred = torch.from_numpy(mask_np.astype(np.float32)).to(device)

                # Save visualizations at intervals
                if (i+1) % viz_interval == 0:
                    show_true_mask = whole_mask_true.cpu().detach().squeeze().numpy()
                    show_whole_mask_pred = whole_mask_pred.cpu().detach().squeeze().numpy()
                    masks = masks_pred.cpu().detach().numpy()
                    show_whole_mask_pred = np.where(show_whole_mask_pred > 0.5, 1, 0)
                    show_masks_pred = np.where(masks > 0.5, 1, 0)
                    save_visualizations(show_masks_pred, show_whole_mask_pred, show_true_mask, fold_save_dir, i)

                whole_mask_pred = whole_mask_pred.unsqueeze(0).unsqueeze(0)
                whole_mask_true = whole_mask_true.unsqueeze(0)
                dice = dice_coeff(whole_mask_pred, whole_mask_true)
                dice_score_total += dice

                pred_binary = (whole_mask_pred > 0.5).float()
                true_binary = (whole_mask_true > 0.5).float()
                correct_pixels = (pred_binary == true_binary).float().sum()
                total_pixels = pred_binary.numel()
                accuracy = correct_pixels / total_pixels
                segmentation_accuracy_total += accuracy

        # Save metrics
        dice_score = dice_score_total / total_images
        seg_accuracy = segmentation_accuracy_total / total_images
        metrics = {}
        metrics["dice_score"] = dice_score.item() if torch.is_tensor(dice_score) else dice_score
        metrics["seg_accuracy"] = seg_accuracy.item() if torch.is_tensor(seg_accuracy) else seg_accuracy
        with open(metrics_file, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_file}")
        logger.info(f"DICE score: {dice_score:.2%}")
        logger.info(f"Segmentation Accuracy: {seg_accuracy:.2%}")
        logger.info(f"Evaluation completed successfully for fold {args.fold}.")
        logger.info(f"{'-' * 80}")

    except Exception as e:
        logger.error(f"Test failed for fold {args.fold} - {str(e)}")
        traceback.print_exc()
        exit(1)

    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
