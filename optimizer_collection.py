import torch
import wandb
## Torch Optimizer List
## Adadelta, Adagrad, Adam, AdamW, SparseAdam, Adamax, ASGD, LBFGS, NAdam, RAdam, RMSprop, Rprop, SGD
def Adam(model):
    return torch.optim.Adam(params = model.parameters(), lr = wandb.config.learning_rate, weight_decay=1e-6)

def AdamW(model):
    return torch.optim.AdamW(params = model.parameters(), lr = wandb.config.learning_rate,weight_decay=1e-6)

def RMSprop(model):
    return torch.optim.RMSprop(params = model.parameters(), lr = wandb.config.learning_rate, weight_decay=1e-6)