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
from dataset import generate_dataset
from data_preparation import create_split
from model import create_model

from utils.utils import *
from tqdm.auto import tqdm

import torch_directml
device = torch_directml.device()


# Ensure deterministic behavior
torch.backends.cudnn.deterministic = True
random.seed(hash("setting random seeds") % 2**32 - 1)
np.random.seed(hash("improves reproducibility") % 2**32 - 1)
torch.manual_seed(hash("by removing stochasticity") % 2**32 - 1)
torch.cuda.manual_seed_all(hash("so runs are repeatable") % 2**32 - 1)

# Device configuration
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def model_pipeline(cfg:dict) -> None:
    # tell wandb to get started
    with wandb.init(project="caption", notes='execution', tags=['main'], reinit=True, config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        config = wandb.config
        print('GENERATING DATASET')
        train_loader, test_loader, vocab = generate_dataset()
        print('DATALOADER CREATED')
        model, criterion, optimizer = create_model(
            embed_size=300,
            attention_dim=256,
            encoder_dim=2048,
            decoder_dim=512,
            vocab = vocab,
            learning_rate=3E-4
        )

        # and use them to train the model
        train(model, optimizer, criterion, cfg['epochs'], train_loader, vocab)
        
        save_model(model, cfg['epochs'], embed_size=300,attention_dim=256,encoder_dim=2048,decoder_dim=512, vocab_size=len(vocab))
        # and test its final performance
        captions_reals, captions_predits, images_list = test(model, test_loader, )

    return model




if __name__ == "__main__":
    
    config = {
        'embed_size': 300,
        'attention_dim': 256,
        'encoder_dim': 2048,
        'decoder_dim': 512,
        'learning_rate': 3e-4,
        'epochs': 1,
        #'batch_size': 256
        'batch_size':64
    }

    create_split()
    print('DATA SPLIT DONE')
    model = model_pipeline(config)

