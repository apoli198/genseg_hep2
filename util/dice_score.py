import torch
import torch.nn as nn
from torch import Tensor


class DiceLoss(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, pred, target, smooth=1e-8):
        pred_sig = torch.sigmoid(pred)
        pred_flat = pred_sig.view(-1)
        target_flat = target.view(-1)

        intersection = (pred_flat * target_flat).sum()
        dice_score = (2. * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

        return 1.0 - dice_score
