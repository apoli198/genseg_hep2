import matplotlib
matplotlib.use('Agg')

import os
import sys
sys.path.append('.')
import argparse
import pickle
import logging
import traceback
import torch
import random
import numpy as np
import gc
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader, Subset
import torchvision.transforms.functional as F

from models_pix2pix import create_model, networks
from util.HEp2_loader import TrainingDataset

networks.conv_arch.requires_grad_(False)
networks.upconv_arch.requires_grad_(False)

def load_fold_splits(dataset, fold):
    """Load train/validation indices for a specific fold"""
    df = dataset.df
    
    train_idx = df.index[df['fold'] != fold].tolist()
    val_idx = df.index[df['fold'] == fold].tolist()
    
    return train_idx, val_idx

# def augment(masks, images=None, angles=[0, 90, 180, 270]):
#     """
#     Apply a random rotation (from 'angles') to each sample in the batch.

#     Args:
#         masks (torch.Tensor): Batch of mask tensors [B, C, H, W].
#         images (torch.Tensor, optional): Batch of image tensors [B, C, H, W].
#         angles (list[int]): List of possible rotation angles.

#     Returns:
#         If images is provided: (aug_images, aug_masks)
#         If not: aug_masks
#     """
#     aug_masks = []
#     aug_images = [] if images is not None else None

#     for i in range(masks.shape[0]):
#         angle = random.choice(angles)

#         # rotate mask
#         mask = masks[i]
#         aug_masks.append(F.rotate(mask, angle))

#         # rotate image if provided
#         if images is not None:
#             img = images[i]
#             aug_images.append(F.rotate(img, angle))

#     aug_masks = torch.stack(aug_masks)
#     if images is not None:
#         aug_images = torch.stack(aug_images)
#         return aug_images, aug_masks
#     return aug_masks

def main():
    train_losses_G = []
    train_losses_D = []
    val_losses_G = []
    val_losses_D = []

    # ---------------------------
    # Parse training options
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--opt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()
    with open(args.opt_path, "rb") as f:
        opt = pickle.load(f)

    # ---------------------------
    # Set up directories
    # ---------------------------
    data_dir = opt.dataroot
    plot_path = os.path.join("./plots/pix2pix", f'losses_fold{args.fold}.png')
    visual_path = os.path.join("./visuals", "pix2pix")
    metrics_file = f"{args.save_path}-fold{args.fold}/metrics.pkl"

    # ---------------------------
    # Set up logging
    # ---------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # ---------------------------
    # Set up model
    # ---------------------------
    model = create_model(opt)
    model.setup(opt)

    # ---------------------------
    # Create dataset and subsets
    # ---------------------------
    dataset = TrainingDataset(data_dir, scale=1.0)

    train_idx, val_idx = load_fold_splits(dataset, args.fold)

    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        loader_args = dict(batch_size=opt.batch_size, num_workers=8, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, **loader_args, drop_last=True)
        val_loader = DataLoader(val_set, shuffle=False, **loader_args, drop_last=True)

        # ---------------------------
        # Main loop over epochs
        # ---------------------------
        logger.info(f'##### Starting Pix2Pix training for fold {args.fold} #####')
        logger.info(f"Training on {len(train_set)} images, validating on {len(val_set)} images")

        for epoch in range(opt.n_epochs):
            model.update_learning_rate()

            # ---------------------------
            # Training
            # ---------------------------
            train_loss_G = 0.0
            train_loss_D = 0.0

            for i, data in enumerate(train_loader):
                train_images = data['image_pix2pix']
                train_masks = data['mask_pix2pix']

                # train_images, train_masks = augment(masks=train_masks, images=train_images)

                model.set_input(train_images, train_masks)

                loss_G, loss_D = model.optimize_parameters()
                train_loss_G += loss_G
                train_loss_D += loss_D

                if (i + 1) % (len(train_loader) // 4) == 0:
                    model.compute_visuals()
                    visuals = model.get_current_visuals()

                    real_image = visuals['pix2pix_real_image'][0].add_(1.0).mul_(0.5).mul_(255.0).add_(0.5).clamp_(0, 255)
                    real_image = real_image.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    fake_image = visuals['pix2pix_fake_image'][0].add_(1.0).mul_(0.5).mul_(255.0).add_(0.5).clamp_(0, 255)
                    fake_image = fake_image.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                    real_mask = visuals['pix2pix_real_mask'][0].mul_(255.0).clamp_(0, 255)
                    real_mask = real_mask.permute(1, 2, 0).to('cpu', torch.uint8).numpy()

                    fig = plt.figure(figsize=(14, 14))
                    fig.suptitle(f'Fold {args.fold} - Epoch {epoch} - Iteration {i + 1}', fontsize=16)

                    gs = fig.add_gridspec(2, 2)

                    ax1 = fig.add_subplot(gs[0, 0])
                    ax1.imshow(real_image, cmap='gray')
                    ax1.set_title('Real Image')
                    ax1.axis('off')

                    ax2 = fig.add_subplot(gs[0, 1])
                    ax2.imshow(fake_image, cmap='gray')
                    ax2.set_title('Fake Image')
                    ax2.axis('off')

                    ax3 = fig.add_subplot(gs[1, :])
                    ax3.imshow(real_mask, cmap='gray')
                    ax3.set_title('Real Mask')
                    ax3.axis('off')

                    plt.tight_layout()

                    visual_save_path = os.path.join(visual_path, f'fold{args.fold}_epoch{epoch}_iter{i + 1}.png')
                    plt.savefig(visual_save_path)
                    plt.close(fig)

            logger.info(f"{'-' * 80}")
            logger.info(f"[Epoch {epoch}] losses:")
            avg_loss_G = train_loss_G / len(train_loader)
            avg_loss_D = train_loss_D / len(train_loader)
            logger.info(f' - Training Loss G: {avg_loss_G:.4f}, D: {avg_loss_D:.4f}')

            train_losses_G.append(avg_loss_G)
            train_losses_D.append(avg_loss_D)

            # ---------------------------
            # Validation
            # ---------------------------
            model.eval()
            val_loss_G = 0.0
            val_loss_D = 0.0
            for val_data in val_loader:
                valid_images = val_data['image_pix2pix']
                valid_masks = val_data['mask_pix2pix']
                loss_G_val, loss_D_val = model.evaluate(image=valid_images, mask=valid_masks)
                val_loss_G += loss_G_val
                val_loss_D += loss_D_val

            avg_val_loss_G = val_loss_G / len(val_loader)
            avg_val_loss_D = val_loss_D / len(val_loader)
            logger.info(f' - Validation Loss G: {avg_val_loss_G:.4f}, D: {avg_val_loss_D:.4f}')

            val_losses_G.append(avg_val_loss_G)
            val_losses_D.append(avg_val_loss_D)

            model.netG.train()
            model.netD.train()

        logger.info(f"{'-' * 80}")
        logger.info("Saving final results...")

        fig, ax1 = plt.subplots(figsize=(20, 12))
        color_train = 'tab:red'
        color_val = 'tab:orange'
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Generator Loss')
        ax1.plot(train_losses_G, label='Training Generator Loss', marker='o', color=color_train)
        ax1.plot(val_losses_G, label='Validation Generator Loss', marker='o', color=color_val)
        ax1.tick_params(axis='y')
        ax1.legend(loc='upper left')
        ax2 = ax1.twinx()
        color_train = 'tab:blue'
        color_val = 'tab:green'
        ax2.set_ylabel('Discriminator Loss')
        ax2.plot(train_losses_D, label='Training Discriminator Loss', marker='o', color=color_train)
        ax2.plot(val_losses_D, label='Validation Discriminator Loss', marker='o', color=color_val)
        ax2.tick_params(axis='y')
        ax2.legend(loc='upper right')
        fig.tight_layout()  
        plt.title('Generator and Discriminator Losses')
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Plot saved to {plot_path}")
            
        model.save_model(f'{args.save_path}-fold{args.fold}')
        best_metrics = {}
        best_metrics['val_G'] = avg_val_loss_G
        best_metrics['val_D'] = avg_val_loss_D
        with open(metrics_file, 'wb') as f:
            pickle.dump(best_metrics, f)
        logger.info(f'Metrics saved to {metrics_file}')
        logger.info(f'##### Training completed for fold {args.fold} #####')

    except Exception as e:
        logger.error(f"Training failed for fold {args.fold} - {str(e)}")
        traceback.print_exc()
        exit(1)

    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == '__main__':
    main()
