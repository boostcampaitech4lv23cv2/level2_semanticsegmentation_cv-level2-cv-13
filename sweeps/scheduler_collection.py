import torch.optim.lr_scheduler as lr_scheduler
import wandb

def CosineAnnealingWarmRestarts(optimizer):
    return lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer,T_0=5, T_mult=2,eta_min=0)