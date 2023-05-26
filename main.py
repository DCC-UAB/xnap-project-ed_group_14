import os
import random
import wandb

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

from train import *
from test import *
from dataset import get_data_loader
from utils.utils import *
from tqdm.auto import tqdm

# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="caption", notes='execution', tags=['main'], reinit=True, config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config

        train_loader, test_loader = 

        # and use them to train the model
        train(model, optimizer, criterion, epochs, data_loader)
        break;
        # and test its final performance
        test(model, test_loader)

    return model




if __name__ == "__main__":
    
    config = {
        'embed_size': 300,
        'vocab_size' : len(dataset.vocab),
        'attention_dim': 256,
        'encoder_dim': 2048,
        'decoder_dim': 512,
        'learning_rate': 3e-4,
        'epochs': 2,
        'batch_size': BATCH_SIZE
    }
        
    model = model_pipeline(config)

