import torch
import torch.nn as nn
from torch.cuda.amp import autocast

def evaluate(net, dataloader, device, criterion_dice, amp=False):
    net.eval()

    running_loss = 0.0
    with torch.no_grad():
        for batch in dataloader:
            images = batch['image'].to(device, dtype=torch.float32)
            masks = batch['mask'].to(device, dtype=torch.float32)

            with autocast(enabled=amp):
                masks_pred = net(images)
                loss = criterion_dice(masks_pred, masks)
                running_loss += loss.item()

    net.train()
    return running_loss / len(dataloader)