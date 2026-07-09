from email.mime import image
import torch
from .base_model import BaseModel
from . import networks
import sys
import os
import torchvision.transforms as transforms
from torch.cuda.amp import autocast, GradScaler

class Pix2PixModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=50.0, help='weight for L1 loss')
            parser.add_argument('--lambda_L2', type=float, default=500.0, help='weight for L2 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_mask', 'fake_image', 'real_image']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D']
        else:  # during test time, only load G
            self.model_names = ['G']
        # init scaler for AMP
        self.scaler = GradScaler(enabled=opt.amp)
        # define networks (both generator and discriminator)
        networks.conv_arch = torch.nn.Parameter(networks.conv_arch.to(self.device))
        networks.upconv_arch = torch.nn.Parameter(networks.upconv_arch.to(self.device))
        self.netG = networks.define_G(input_nc=opt.input_nc, output_nc=opt.output_nc, ngf=opt.ngf, netG=opt.netG, norm=opt.norm,
                                      use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        if self.isTrain:
            self.netD = networks.define_D(input_nc=opt.input_nc + opt.output_nc, ndf=opt.ndf, netD=opt.netD, n_layers_D=opt.n_layers_D,
                                          norm=opt.norm, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode, target_real_label=0.9, target_fake_label=0.1).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-5)
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(0.5, 0.999), weight_decay=1e-5)
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, image, mask):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.real_mask = mask.to(device=self.device, dtype=torch.float32)
        self.real_image = image.to(device=self.device, dtype=torch.float32)

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        with autocast(enabled=self.opt.amp):
            self.fake_image = self.netG(self.real_mask)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        with autocast(enabled=self.opt.amp):
            # Fake; stop backprop to the generator by detaching fake_B
            fake_mask_image = torch.cat((self.real_mask, self.fake_image.detach()), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
            pred_fake = self.netD(fake_mask_image)
            self.loss_D_fake = self.criterionGAN(pred_fake, False)
            # Real
            real_mask_image = torch.cat((self.real_mask, self.real_image), 1)
            pred_real = self.netD(real_mask_image)
            self.loss_D_real = self.criterionGAN(pred_real, True)
            # combine loss and calculate gradients
            self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.scaler.scale(self.loss_D).backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        with autocast(enabled=self.opt.amp):
            # First, G(A) should fake the discriminator
            fake_mask_image = torch.cat((self.real_mask, self.fake_image), 1)
            pred_fake = self.netD(fake_mask_image)
            self.loss_G_GAN = self.criterionGAN(pred_fake, True)
            # Second, G(A) = B
            self.loss_G_L1 = self.criterionL1(self.fake_image, self.real_image) * self.opt.lambda_L1
            self.loss_G_L2 = self.criterionL2(self.fake_image, self.real_image) * self.opt.lambda_L2
            # combine loss and calculate gradients
            self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_G_L2
        self.scaler.scale(self.loss_G).backward()
  
    def optimize_parameters(self):
        self.forward()                              # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)     # enable backprop for D
        self.optimizer_D.zero_grad()            # set D's gradients to zero
        self.backward_D()         # calculate gradients for D
        self.scaler.step(self.optimizer_D)
        self.scaler.update()                    # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)    # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()            # set G's gradients to zero
        self.backward_G()                           # calculate graidents for G
        self.scaler.step(self.optimizer_G)
        self.scaler.update()                    # update G's weights
        return self.loss_G.item(), self.loss_D.item()

    def evaluate(self, image, mask):
        """
        Run the model on a validation batch and compute generator and discriminator losses.

        Returns:
            loss_G (float): Total generator loss.
            loss_D (float): Total discriminator loss.
        """
        with torch.no_grad(), autocast(enabled=self.opt.amp):
            real_mask = mask.to(self.device, dtype=torch.float32)
            real_image = image.to(self.device, dtype=torch.float32)
            # Generate fake images
            fake_image = self.netG(real_mask)
            fake_mask_image = torch.cat((real_mask, fake_image), 1)
            real_mask_image = torch.cat((real_mask, real_image), 1)
            # Discriminator predictions
            pred_fake = self.netD(fake_mask_image)
            pred_real = self.netD(real_mask_image)
            # Generator losses
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            loss_G_L1 = self.criterionL1(fake_image, real_image) * self.opt.lambda_L1
            loss_G_L2 = self.criterionL2(fake_image, real_image) * self.opt.lambda_L2
            loss_G = loss_G_GAN + loss_G_L1 + loss_G_L2
            # Discriminator losses
            loss_D_fake = self.criterionGAN(pred_fake, False)
            loss_D_real = self.criterionGAN(pred_real, True)
            loss_D = (loss_D_fake + loss_D_real) * 0.5

            return loss_G.item(), loss_D.item()

    def save_model(self, save_path):
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        torch.save(self.netD.state_dict(), save_path+'/pix2pix_discriminator.pkl')
        torch.save(self.netG.state_dict(), save_path+'/pix2pix_generator.pkl')

    def load_model(self, D_model_filename, G_model_filename):
        D_model_path = os.path.join(os.getcwd(), D_model_filename)
        G_model_path = os.path.join(os.getcwd(), G_model_filename)
        self.netD.load_state_dict(torch.load(D_model_path, map_location={'cuda:0':'cuda:0'}))
        self.netG.load_state_dict(torch.load(G_model_path, map_location={'cuda:0':'cuda:0'}), strict=False)
