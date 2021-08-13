from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler

# to make import from one dir up
import sys
sys.path.append("..") 

# import the custom Dataset object
import SMCCarsDataset

#import model
from model import UNet

import torch
import torch.nn as nn
import numpy as np

#import for multi-gpu training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp


def create_data_loader(rootdir, rank, world_size, batch_size, dataType = ''):
    """
    Desscription: Creates training and testing data using the SMCCarsDataset class and Pytorch's Dataloader. 
    Args:
        rootdir (string): directory where the data is stored
        rank (int): process ID for the GPU
        world_size (int): total number of GPUs
        batch_size (int): size of each batch
        dataType (string): must be either 'train' or 'test' for training and testing data, respectively
    """


    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir, traditional_transform=True, overlay_transform=True)
    
    #set seed for reproducibility
    torch.manual_seed(99)

    #random 80/20 split of data into training and testing, respectively
    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())

    validDataTypesNames = ['train', 'test']
    validDataTypes = {'train':train_data, 'test':test_data}

    if dataType not in validDataTypesNames:
        sys.exit('In create_data_loaders, there must be a valid dataType.')
    
    #create and return data loaders
    dataSampler = DistributedSampler(validDataTypes[dataType], num_replicas = world_size, rank=rank, shuffle=False, seed = 99)
    dataLoader = DataLoader(validDataTypes[dataType], batch_size = batch_size, sampler = dataSampler, pin_memory=True)

    return dataLoader

def create_model(rank):
    """
    Desscription: Creates model using Pytorch's DistributedDataParallel training strategy (DDP) and model.py
    Args:
        rank (int): process ID for the GPU
    """

    device = torch.device(f'cuda:{rank}')

    #model from main.py
    model = UNet()
    model = model.to(device)
    
    #use DDP
    model = DDP(model, device_ids=[rank], output_device=rank)

    #define optimizers and loss function
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()

    return model, opt, loss_func