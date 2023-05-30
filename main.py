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
from prediction import predict

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
    with wandb.init(project="caption", name=cfg.get('execution_name'),notes='execution', tags=['main'], reinit=True, config=cfg):
        # access all HPs through wandb.config, so logging matches execution!
        wandb.define_metric('loss_train',step_metric='epoch')
        wandb.define_metric('loss_test',step_metric='epoch')

        wandb.define_metric('trai_bleu', step_metric='epoch')
        wandb.define_metric('test_bleu', step_metric='epoch')

        wandb.define_metric('perp_train', step_metric='epoch')
        wandb.define_metric('perp_test', step_metric='epoch')

        

        print('GENERATING DATASET')
        train_loader, test_loader, vocab = generate_dataset(cfg.get('batch_size'))
        
        print('DATALOADER CREATED')

        model, criterion, optimizer = create_model(
            embed_size=cfg.get('embed_size'),
            attention_dim=cfg.get('attention_dim'),
            encoder_dim=cfg.get('encoder_dim'),
            decoder_dim=cfg.get('decoder_dim'),
            vocab = vocab,
            learning_rate=cfg.get('learning_rate'),
            optimizer_type=cfg.get('optimizer_type')
        )

        # and use them to train the model
        train(model, optimizer, criterion, cfg['epochs'], train_loader, vocab, test_loader)
        print('MAKING SOME PREDICTIONS')
        predict(test_loader, model, train_loader.dataset.vocab)
        
        print('SAVING MODEL')        
        save_model(model=model, 
                   num_epochs=cfg['epochs'], 
                   embed_size=cfg.get('embed_size'), 
                   attention_dim=cfg.get('attention_dim'), 
                   encoder_dim=cfg.get('encoder_dim'), 
                   decoder_dim=cfg.get('decoder_dim'), 
                   vocab_size=len(vocab),
                   name=cfg.get('execution_name'))
        
        # and test its final performance
        #captions_reals, captions_predits, images_list = test(model, test_loader, )

    return model




if __name__ == "__main__":
    
    config = {
        'embed_size': 1024,
        'attention_dim': 1024,
        'encoder_dim': 4096,
        'decoder_dim': 1024,
        'learning_rate': 3e-4,
        'epochs': 15,
        'batch_size':256,
        'execution_name':'azure-max-values',
        'optimizer_type':'Adadelta',
    }

    create_split()
    print('DATA SPLIT DONE')
    model = model_pipeline(config)

    