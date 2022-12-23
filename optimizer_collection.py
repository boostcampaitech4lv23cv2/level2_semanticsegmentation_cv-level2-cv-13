import torch
import wandb

def Adam(model):
    return torch.optim.Adam(params = model.parameters(), lr = wandb.config.learning_rate, weight_decay=1e-6)