import gc
import math
import os
import sys
sys.path.append('.')
import traceback
import pickle
import logging
import argparse
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast, GradScaler
import pandas as pd
import matplotlib.pyplot as plt

from util.HEp2_loader import TrainingDataset
from util.dice_score import DiceLoss
from unet import UNet
from unet.evaluate import evaluate

def load_fold_splits(dataset, fold):
    """Load train/validation indices for a specific fold"""
    df = dataset.df
    
    train_idx = df.index[df['fold'] != fold].tolist()
    val_idx = df.index[df['fold'] == fold].tolist()
    
    return train_idx, val_idx

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight.data, 1.0, init_gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
    
    print('initialize network with %s' % init_type)
    net.apply(init_func)

def main():
    val_best_loss = float('inf')
    best_epoch = 0
    patience = 5
    min_delta = 0.001
    running_loss = 0.0

    # ---------------------------
    # Parse training options
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True, help="Fold index for cross-validation")
    parser.add_argument("--opt_path", type=str, required=True, help="Path to the pickle file with options")
    parser.add_argument("--save_path", type=str, required=True, help="Path to save the trained model")
    args = parser.parse_args()
    with open(args.opt_path, "rb") as f:
        opt = pickle.load(f)

    # ---------------------------
    # Set up device and directories
    # ---------------------------
    assert opt.cuda_index == int(opt.gpu_ids[0]), "GPU indices should be the same"
    device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')
    data_dir = opt.dataroot
    fold_save_dir = args.save_path+'-fold'+str(args.fold)
    if not os.path.exists(fold_save_dir):
        os.makedirs(fold_save_dir)
    unet_save_path = os.path.join(fold_save_dir, str(opt.seg_model)+'.pkl')
    plot_path = os.path.join("./plots/baseline", f'losses_fold{args.fold}.png')
    metrics_file = os.path.join(fold_save_dir, 'metrics.pkl')

    # ---------------------------
    # Set up logging
    # ---------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)

    # ---------------------------
    # Create segmentation model (UNet)
    # ---------------------------
    if opt.seg_model.lower() == 'unet':
        net = UNet(n_channels=opt.output_nc, n_classes=opt.classes, bilinear=True)
    else:
        logger.error(f"Unknown segmentation model option: {opt.seg_model}")
        sys.exit(1)
    init_weights(net, init_type=opt.init_type, init_gain=opt.init_gain)
    net = net.to(device)

    # ---------------------------
    # Define optimizer and scheduler for segmentation network
    # ---------------------------
    optimizer_unet = optim.Adam(net.parameters(), lr=opt.unet_learning_rate)
    criterion = DiceLoss().to(device)
    scaler = GradScaler(enabled=opt.amp)

    # ---------------------------
    # Create dataset and subsets
    # ---------------------------
    dataset = TrainingDataset(data_dir, 1.0)
    train_idx, val_idx = load_fold_splits(dataset, args.fold)
    train_set = Subset(dataset, train_idx)
    val_set = Subset(dataset, val_idx)

    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        loader_args = dict(batch_size=opt.batch_size, num_workers=8, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        logger.info(f'The number of training images for fold {args.fold} = {len(train_set)}')
        logger.info(f'The number of validation images for fold {args.fold} = {len(val_set)}')

        train_losses = []
        val_losses = []
        for epoch in range(opt.n_epochs):
            net.train()
            running_loss = 0.0

            for batch in train_loader:
                images = batch['image'].to(device=device, dtype=torch.float32)
                masks = batch['mask'].to(device=device, dtype=torch.float32)

                optimizer_unet.zero_grad()
                with autocast(enabled=opt.amp):
                    masks_pred = net(images)
                    loss = criterion(masks_pred, masks)
                scaler.scale(loss).backward()
                scaler.step(optimizer_unet)
                scaler.update()

                running_loss += loss.item()

            logger.info(f"{'-' * 80}")
            logger.info(f"[Epoch {epoch}] losses:")
            train_loss = running_loss / len(train_loader)
            logger.info(f" - Training Loss: {train_loss:.4f}")
            train_losses.append(train_loss)
            val_loss = evaluate(net, val_loader, device, criterion, amp=opt.amp)
            logger.info(f" - Validation Loss: {val_loss:.4f}")
            val_losses.append(val_loss)

            if not math.isnan(val_loss):
                if val_loss < val_best_loss - min_delta:
                    val_best_loss = val_loss
                    best_epoch = epoch
                    torch.save(net.state_dict(), unet_save_path)
                    logger.info(f"New best model saved. Val loss: {val_loss:.4f}")
                elif epoch - best_epoch >= patience:
                    logger.info(f'Early stopping triggered at epoch {epoch}')
                    break
                else:
                    logger.info(' ')
            else:
                logger.info(f"Early stopping triggered due to NaN loss at epoch {epoch}")
                break

        logger.info(f"{'-' * 80}")
        logger.info("Saving final results...")

        plt.figure(figsize=(20, 12))
        plt.plot(train_losses, label='Training Loss', marker='o', color='red')
        plt.plot(val_losses, label='Validation Loss', marker='o', color='orange')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Losses')
        plt.legend()
        plt.grid(True)
        plt.savefig(plot_path)
        plt.close()
        logger.info(f"Plot saved to {plot_path}")

        metrics = {'val_loss': val_best_loss}
        with open(metrics_file, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f'Metrics saved to {metrics_file}')
        best_state_dict = torch.load(unet_save_path)
        net.load_state_dict(best_state_dict)
        torch.save(net.state_dict(), os.path.join(fold_save_dir, 'final.pkl'))
        logger.info(f"Final segmentation model for fold {args.fold} saved to {fold_save_dir}/final.pkl")

    except Exception as e:
        logger.error(f"Training failed for fold {args.fold} - {str(e)}")
        traceback.print_exc()
        exit(1)
        
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
