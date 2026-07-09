import gc
import os
import sys
import traceback
sys.path.append('.')
import pickle
import logging
import argparse
import random
import math
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms.functional as F
from torch.utils.data import DataLoader, Subset
from torch.cuda.amp import autocast

from util.HEp2_loader import TrainingDataset
from util.dice_score import DiceLoss
from models_pix2pix import create_model, networks
from unet import UNet
from unet.evaluate import evaluate

from betty.engine import Engine
from betty.configs import Config, EngineConfig
from betty.problems import ImplicitProblem

networks.conv_arch.requires_grad_(True)
networks.upconv_arch.requires_grad_(True)

show_index = 0
running_loss = 0.0
val_best_loss = float('inf')
epoch = 0
best_epoch = 0
patience = 5       # 5 | 10
min_delta = 0.001

def load_fold_splits(dataset, fold):
    """Load train/validation indices for a specific fold"""
    df = dataset.df
    
    train_idx = df.index[df['fold'] != fold].tolist()
    val_idx = df.index[df['fold'] == fold].tolist()
    
    return train_idx, val_idx

def init_weights(net, init_type='normal', init_gain=0.02):
    """Initialize network weights safely (without using .data)."""
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            with torch.no_grad():
                if init_type == 'normal':
                    nn.init.normal_(m.weight, 0.0, init_gain)
                elif init_type == 'xavier':
                    nn.init.xavier_normal_(m.weight, gain=init_gain)
                elif init_type == 'kaiming':
                    nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
                elif init_type == 'orthogonal':
                    nn.init.orthogonal_(m.weight, gain=init_gain)
                else:
                    raise NotImplementedError(f'initialization method [{init_type}] is not implemented')
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            with torch.no_grad():
                if hasattr(m, 'weight') and m.weight is not None:
                    nn.init.normal_(m.weight, 1.0, init_gain)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    print(f'initialize network with {init_type}')
    net.apply(init_func)

def morph_op(mask, op="none", radius=3):
    """
    Apply morphological operation directly on a tensor mask [C,H,W].
    op ∈ {"none", "erode", "dilate"}.
    """
    if op == "none":
        return mask

    k = 2*radius + 1  # kernel size

    if op == "dilate":
        out = nn.functional.max_pool2d(mask.unsqueeze(0), kernel_size=k, stride=1, padding=radius)
        return out.squeeze(0)

    if op == "erode":
        out = -nn.functional.max_pool2d(-mask.unsqueeze(0), kernel_size=k, stride=1, padding=radius)
        return out.squeeze(0)

    return mask

def augment(masks, angles=[0, 90, 180, 270], morph_radius=3):
    aug_masks = []
    for i in range(masks.shape[0]):
        angle = random.choice(angles)
        mask = F.rotate(masks[i], angle)

        # morph_choice = random.choice(["none", "erode", "dilate"])
        # mask = morph_op(mask, morph_choice, radius=morph_radius)

        aug_masks.append(mask)
    aug_masks = torch.stack(aug_masks)
    return aug_masks
    
def main():
    global val_best_loss, best_epoch, patience, min_delta

    # ---------------------------
    # Parse training options
    # ---------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument("--fold", type=int, required=True)
    parser.add_argument("--opt_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, required=True)
    args = parser.parse_args()

    with open(args.opt_path, 'rb') as f:
        opt = pickle.load(f)

    # ---------------------------
    # Set up device and directories
    # ---------------------------
    assert opt.cuda_index == int(opt.gpu_ids[0]), "GPU indices should be the same"
    device = torch.device(f'cuda:{opt.cuda_index}' if torch.cuda.is_available() else 'cpu')
    data_dir = opt.dataroot
    pretrained_path = os.path.join("./pix2pix_HEp2_model")
    fold_save_dir = args.save_path+'-fold'+str(args.fold)
    if not os.path.exists(fold_save_dir):
        os.makedirs(fold_save_dir)
    unet_save_path = fold_save_dir+'/'+str(opt.seg_model)+'.pkl'
    plot_path = os.path.join("./plots/end2end", f'losses_fold{args.fold}.png')
    visuals_path = os.path.join("./visuals", 'end2end')
    metrics_file = fold_save_dir+'/metrics.pkl'

    # ---------------------------
    # Set up logging
    # ---------------------------
    logging.basicConfig(level=logging.INFO, format='%(asctime)s: %(message)s', stream=sys.stdout)
    logger = logging.getLogger(__name__)

    try:
        # ---------------------------
        # Load pre-trained Pix2Pix model for current fold
        # ---------------------------
        model = create_model(opt)
        model.setup(opt)
        model_path = os.path.join(pretrained_path, f"pix2pix-HEp2-fold{args.fold}")
        model.load_model(model_path + '/pix2pix_discriminator.pkl', model_path + '/pix2pix_generator.pkl')

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
        optimizer_unet = optim.Adam(net.parameters(), lr=opt.unet_learning_rate, betas=(0.9, 0.999),
                                    weight_decay=1e-8)
        scheduler_unet = optim.lr_scheduler.ReduceLROnPlateau(optimizer_unet, mode='min', patience=2)

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

        loader_args = dict(batch_size=opt.batch_size, num_workers=8, pin_memory=True)
        train_loader = DataLoader(train_set, shuffle=True, drop_last=True, **loader_args)
        val_loader = DataLoader(val_set, shuffle=False, drop_last=True, **loader_args)

        logger.info(f'The number of training images for fold {args.fold} = {len(train_set)}')
        logger.info(f'The number of validation images for fold {args.fold} = {len(val_set)}')

        train_iters = opt.n_epochs * len(train_loader)

        # ---------------------------
        # Define loss functions
        # ---------------------------
        criterion_BCE = nn.BCEWithLogitsLoss()
        criterion_dice = DiceLoss().to(device)
        criterion_GAN = networks.GANLoss(opt.gan_mode, target_real_label=0.9, target_fake_label=0.1).to(device)
        criterion_L1 = nn.L1Loss()
        criterion_L2 = nn.MSELoss()

        class Generator(ImplicitProblem):
            def training_step(self, batch):
                images = batch['image_pix2pix'].to(dtype=torch.float32, device=device)
                masks = batch['mask_pix2pix'].to(dtype=torch.float32, device=device)

                with autocast(enabled=opt.amp):
                    fake_images = self.module(masks)
                    fake_pairs = torch.cat((masks, fake_images), 1)
                    pred_fake = self.discriminator(fake_pairs)
                    loss_G_GAN = criterion_GAN(pred_fake, True)
                    loss_G_L1 = criterion_L1(fake_images, images) * opt.lambda_L1
                    loss_G_L2 = criterion_L2(fake_images, images) * opt.lambda_L2
                    loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2

                return loss_G
            
        class Discriminator(ImplicitProblem):
            def training_step(self, batch):
                images = batch['image_pix2pix'].to(dtype=torch.float32, device=device)
                masks = batch['mask_pix2pix'].to(dtype=torch.float32, device=device)

                with autocast(enabled=opt.amp):
                    fake_images = self.generator(masks).detach()
                    fake_pairs = torch.cat((masks, fake_images), 1)
                    pred_fake = self.module(fake_pairs)
                    loss_D_fake = criterion_GAN(pred_fake, False)

                    real_pairs = torch.cat((masks, images), 1)
                    pred_real = self.module(real_pairs)
                    loss_D_real = criterion_GAN(pred_real, True)
                    loss_D = (loss_D_fake + loss_D_real) * 0.5

                return loss_D
            
        class UNetProblem(ImplicitProblem):
            def __init__(self, generator_problem, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.generator_problem = generator_problem
                self._last_logged_count = -1

            def training_step(self, batch):
                global epoch, show_index, running_loss

                is_primary_training_call = (
                    self._training and
                    self._count != self._last_logged_count
                )

                images = batch['image'].to( dtype=torch.float32, device=device)
                masks_cpu = batch['mask'].to(dtype=torch.float32)
                masks = masks_cpu.to(device=device)

                with autocast(enabled=opt.amp):
                    pred_real = self.module(images)
                    real_BCE = criterion_BCE(pred_real, masks)
                    real_dice = criterion_dice(pred_real, masks)
                    loss_real = real_BCE + real_dice

                    fake_masks = augment(masks=masks_cpu)
                    fake_masks = fake_masks.to(device=device)
                    fake_images = self.generator_problem.module(fake_masks)
                    fake_images = fake_images.add(1.0).mul(0.5)

                    pred_fake = self.module(fake_images)
                    fake_BCE = criterion_BCE(pred_fake, fake_masks)
                    fake_dice = criterion_dice(pred_fake, fake_masks)
                    loss_fake = fake_BCE + fake_dice

                    loss = loss_fake + opt.loss_lambda * loss_real
                    running_loss += loss.item()

                if is_primary_training_call:
                    self._last_logged_count = self._count
                    show_index += 1
                    if show_index % (len(train_loader) // 4) == 0:
                        show_image = images[0].mul(255).add(0.5).clamp(0, 255)
                        show_image = show_image.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        show_mask = masks[0].mul(255).clamp(0, 255)
                        show_mask = show_mask.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        show_fake_image = fake_images[0].mul(255).add(0.5).clamp(0, 255)
                        show_fake_image = show_fake_image.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        show_fake_mask = fake_masks[0].mul(255).clamp(0, 255)
                        show_fake_mask = show_fake_mask.permute(1, 2, 0).to('cpu', torch.uint8).numpy()
                        
                        fig = plt.figure(figsize=(14, 14))
                        fig.suptitle(f'Fold {args.fold} - Epoch {epoch} - Iteration {show_index}', fontsize=16)
                        
                        gs = fig.add_gridspec(2, 2)
                        
                        ax1 = fig.add_subplot(gs[0, 0])
                        ax1.imshow(show_image, cmap='gray')
                        ax1.set_title('Real Image')
                        ax1.axis('off')
                        
                        ax2 = fig.add_subplot(gs[0, 1])
                        ax2.imshow(show_fake_image, cmap='gray')
                        ax2.set_title('Fake Image')
                        ax2.axis('off')
                        
                        ax3 = fig.add_subplot(gs[1, 0])
                        ax3.imshow(show_mask, cmap='gray')
                        ax3.set_title('Real Mask')
                        ax3.axis('off')
                        
                        ax4 = fig.add_subplot(gs[1, 1])
                        ax4.imshow(show_fake_mask, cmap='gray')
                        ax4.set_title('Fake Mask')
                        ax4.axis('off')
                        
                        plt.tight_layout()
                        
                        visual_save_path = os.path.join(visuals_path, f'fold{args.fold}_epoch{epoch}_iter{show_index}.png')
                        plt.savefig(visual_save_path)
                        plt.close(fig)

                return loss
            
        class Arch(ImplicitProblem):
            def __init__(self, seg_problem, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.seg_problem = seg_problem

            def training_step(self, batch):
                images = batch['image'].to(device=device, dtype=torch.float32)
                masks = batch['mask'].to(device=device, dtype=torch.float32)

                with autocast(enabled=opt.amp):
                    pred = self.seg_problem.module(images)
                    loss_BCE = criterion_BCE(pred, masks)
                    loss_dice = criterion_dice(pred, masks)
                    loss = loss_BCE + loss_dice

                return loss
            
        train_losses = []
        val_losses = []
        class SSEngine(Engine):
            @torch.no_grad()
            def validation(self):
                global val_best_loss, best_epoch, patience, epoch, running_loss, show_index
                logger.info(f"{'-' * 80}")
                train_loss = running_loss / len(train_loader)
                running_loss = 0.0
                logger.info(f"[Epoch {epoch}] losses:")
                logger.info(f" - Training Loss: {train_loss:.4f}")
                train_losses.append(train_loss)
                val_loss = evaluate(net, val_loader, device, criterion_dice, amp=opt.amp)
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
                        raise StopIteration('Early stopping triggered')
                    else:
                        logger.info(' ')
                else:
                    logger.info(f"Early stopping triggered due to NaN loss at epoch {epoch}")
                    raise StopIteration('Nan loss')

                if self.global_step % len(train_loader) == 0 and self.global_step:
                    logger.info(f"Epoch {epoch} completed")
                    epoch = self.global_step // len(train_loader)
                    show_index = 0
                    scheduler_unet.step(val_loss)

        outer_config = Config(retain_graph=True, fp16=opt.amp)
        inner_config = Config(type="darts", unroll_steps=opt.unroll_steps, fp16=opt.amp)
        engine_config = EngineConfig(
            valid_step=len(train_loader),
            train_iters=train_iters,
            roll_back=True
        )

        generator = Generator(
            name='generator',
            module=model.netG,
            optimizer=model.optimizer_G,
            train_data_loader=train_loader,
            config=inner_config,
            device=device
        )
        
        discriminator = Discriminator(
            name='discriminator',
            module=model.netD,
            optimizer=model.optimizer_D,
            train_data_loader=train_loader,
            config=inner_config,
            device=device
        )

        seg = UNetProblem(
            generator_problem=generator,
            name='seg',
            module=net,
            optimizer=optimizer_unet,
            train_data_loader=train_loader,
            config=inner_config,
            device=device
        )

        optimizer_arch = torch.optim.Adam([model.netG.module.conv_arch, model.netG.module.upconv_arch], lr=opt.arch_lr,
                                          betas=(0.5, 0.999), weight_decay=1e-5)
        arch = Arch(
            seg_problem = seg,
            name='arch',
            module=model.netG,
            optimizer=optimizer_arch,
            train_data_loader=val_loader,
            config=outer_config,
            device=device
        )

        # -----------------------------
        # Run the engine for segmentation training
        # -----------------------------
        problems = [discriminator, generator, seg, arch]
        l2u = {generator: [seg], seg: [arch]}
        u2l = {arch: [generator]}
        dependencies = {"l2u": l2u, "u2l": u2l}

        engine = SSEngine(config=engine_config, problems=problems, dependencies=dependencies)

        try:
            engine.run()
        except StopIteration as e:
            logger.info(e)

        # Save final results
        logger.info(f"{'-' * 80}")
        logger.info("Saving final results...")

        # Plot training and validation losses
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

        # Save final segmentation model
        metrics= {'val_loss': val_best_loss}
        with open(metrics_file, "wb") as f:
            pickle.dump(metrics, f)
        logger.info(f'Metrics saved to {metrics_file}')
        best_state_dict = torch.load(unet_save_path)
        net.load_state_dict(best_state_dict)
        torch.save(net.state_dict(), os.path.join(fold_save_dir, 'final.pkl'))
        logger.info(f"Final segmentation model for fold {args.fold} saved to {unet_save_path}")

    except Exception as e:
        logger.error(f"End-to-end training failed for fold {args.fold}: {str(e)}")
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        torch.cuda.empty_cache()
        gc.collect()

if __name__ == "__main__":
    main()
