# File: main.py
# Author: Gerson Kroiz
# Description: uses model from model.py and
# creates train/test data, trains model

import matplotlib.pyplot as plt
import numpy as np
import os
import GPUtil

# to make import from one dir up
import sys
sys.path.append("..") 
# import the custom Dataset object
import SMCCarsDataset


from torch.utils.data import DataLoader
from torch.utils.data import DistributedSampler
from torchvision.utils import save_image



#import model
from model import UNet

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from collections import defaultdict
from torchsummary import summary

#import for multi-gpu training
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.distributed import Backend
import argparse

from PIL import Image
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode
import torchvision.transforms as transforms

loader = transforms.Compose([
    # transforms.Resize((720, 1280), InterpolationMode.NEAREST),  # scale imported image
    transforms.ToTensor()])

def image_loader(image_name):
    image = Image.open(image_name).convert('RGB')
    image = loader(image)

    if (image.shape[1] == 1024):
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1820))
        image = TF.crop(image, i, j, h, w)
        # NEAREST Interpolation so that segmap is logically interpolated
        resize = transforms.Resize((720, 1280), InterpolationMode.NEAREST)
        image = resize(image)

    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0)

    # image = loader(image).unsqueeze(0)
    return image.to(device, torch.float)

#basic training function
def training(rank, numEpochs, batchSize, dataLoader, model, opt, loss_func, save_model_loc):
    print('in training')
    model.train()
    for epoch in range(numEpochs):
        startEpoch = time.time()
        dataLoader.sampler.set_epoch(epoch)
        epochLoss = 0.0
        numBatches = 0
        for (batch_idx, batch) in enumerate(dataLoader):
            numBatches += 1
            #define input and labels
            inputs = batch['image']
            # image = image.float()
            # segmentation = segmentation.float()

            labels = batch['segmentation']
            
            #put inputs and labels on gpu if permitted
            inputs = inputs.to(device, dtype=torch.float)

            labels = labels.squeeze(axis=1)
            labels = labels.to(device, dtype=torch.int64)


            #print model summary
            if batch_idx == 0 and epoch == 0 and rank == 0:
                print(summary(model, input_size=(3, 1280, 720)))
            #output from model
            out = model(inputs)
            #calculate loss
            # loss = loss_func(torch.sigmoid(out), torch.sigmoid(labels))

            # out = out.long()
            # labels = labels.long()
            # out = out.cuda()
            # labels = torch.cuda.LongTensor(labels)#.long().to(device)
            loss = loss_func(out, labels)
            # print('rank: ' + str(rank) + ', after batch idx: ' + str(batch_idx) + ', after loss1: ' + str(loss))
            # Backward
            opt.zero_grad()
            loss.backward()
            epochLoss += loss.item() * batchSize
            # print('rank: ' + str(rank) + ', after batch idx: ' + str(batch_idx) + ', after loss2: ' + str(loss))

            opt.step()

        endEpoch = time.time()
        if rank == 0:
            print('Epoch: '+str(epoch)+'/'+str(numEpochs)+', Loss: '+str(epochLoss)+', Avg Loss: '+str(epochLoss /(numBatches * batchSize))+', Time: '+str(endEpoch-startEpoch))
    if rank == 0:
        print('before saved model')
        torch.save(model.state_dict(), save_model_loc)
        print("saved model")
    print('rank: ' + str(rank) + ', training done')

def create_data_loaders(rootdir, rank, world_size, batchSize):
    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    torch.manual_seed(99)
    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())
    trainDataSampler = DistributedSampler(train_data, num_replicas = world_size, rank=rank, shuffle=False, seed = 99)
    testDataSampler = DistributedSampler(test_data, num_replicas = world_size, rank=rank, shuffle=False, seed = 99)

    trainDataLoader = DataLoader(train_data, batch_size = batchSize, sampler = trainDataSampler, pin_memory=True)
    testDataLoader = DataLoader(test_data, batch_size = batchSize, sampler = testDataSampler, pin_memory=True)

    return trainDataLoader, testDataLoader


def create_model(rank):
    
    device = torch.device(f'cuda:{rank}')
    model = UNet()
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)

    #new change lr=0.001
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_func = nn.CrossEntropyLoss()
    return model, opt, loss_func


if __name__ == "__main__":

    print('program started')

    #load data
    rootdir = '/raid/gkroiz1/SMC21_GM_AV'
    # rootdir = '/raid/gkroiz1/SMC21_GM_AV_w_style'
    save_model_loc = 'default-model.pt'
    # save_model_loc = 'style-model.pt'
    numEpochs = 25
    batchSize = 5



    #check for GPUs or CPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()


    print('number of cuda devices: ' + str(n_gpus))

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()


    rank = args.local_rank
    print('rank: ' + str(rank))
    worldSize = torch.cuda.device_count()

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    trainDataLoader, testDataLoader = create_data_loaders(rootdir, rank, worldSize, batchSize)
    if rank == 0:
        print('after creating dataLoaders: ')
        GPUtil.showUtilization()

    model, opt, loss_func = create_model(rank)
    if rank == 0:
        print('after creating model: ')
        GPUtil.showUtilization()

    #call training function
    training(rank, numEpochs, batchSize, trainDataLoader, model, opt, loss_func, save_model_loc)








