# File: main.py
# Author: Gerson Kroiz
# Description: uses model from model.py and
# creates train/test data, trains model, and creates
# gt and pred directories for evaluation

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


#basic training function
def training(rank, numEpochs, dataLoader, model, opt, loss_func):
    print('in training')
    model.train()
    for epoch in range(numEpochs):
        startEpoch = time.time()
        dataLoader.sampler.set_epoch(epoch)
    
        for (batch_idx, batch) in enumerate(dataLoader):

            #define input and labels
            inputs = batch['image']
            labels = batch['segmentation']
            
            #put inputs and labels on gpu if permitted
            inputs = inputs.to(device)

            labels = labels.squeeze(axis=1)
            labels = labels.to(device, dtype=torch.int64)


            #print model summary
            # if batch_idx == 0 and epoch == 0 and rank == 0:
            #     print(summary(model, input_size=(3, 1280, 720)))
            #output from model

            # if rank == 0:
                # print('after model summary: ')
                # GPUtil.showUtilization()
            out = model(inputs)
            # if rank == 0:
                # print('after model usage: ')
                # GPUtil.showUtilization()
            #calculate loss
            # loss = loss_func(torch.sigmoid(out), torch.sigmoid(labels))

            # out = out.long()
            # labels = labels.long()
            # out = out.cuda()
            labels = torch.cuda.LongTensor(labels)#.long().to(device)
            loss = loss_func(out, labels)

            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()
        endEpoch = time.time()
        print(f"\nEpoch: {epoch}/{numEpochs}, Loss: {loss}, 'Time: {endEpoch-startEpoch}")
    torch.save(model.state_dict(), 'model.pt')
    print("saved model")


#function for evaluating model
def predictions(dataLoader, model, batchSize):

    #convert model to eval mode
    model = model.eval()

    #create gt and pred directories

    if not os.path.isdir('results'):
        os.mkdir('results')
    os.chdir('results')
    if not os.path.isdir('gt'):
        os.mkdir('gt')
    if not os.path.isdir('pred'):
        os.mkdir('pred')

    #do predictions based on batch size
    for (batch_idx, batch) in enumerate(dataLoader):
        inputs = batch['image']
        labels = batch['segmentation']
        inputs = inputs.to(device)
        labels = labels.to(device)
        predictions = model(inputs)

        #save predictions and ground truth in respective directories
        for index in range(batchSize):
            save_image(predictions[index], 'pred/image' + str(batch_idx * batchSize + index) + '.png')
            save_image(labels[index], 'gt/image' + str(batch_idx * batchSize + index) + '.png')

        
def create_data_loaders(rank, world_size, batchSize):
    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)
    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())
    trainDataSampler = DistributedSampler(train_data, num_replicas = world_size, rank=rank, seed = 99)
    testDataSampler = DistributedSampler(train_data, num_replicas = world_size, rank=rank, seed = 99)

    trainDataLoader = DataLoader(train_data, batch_size = batchSize, shuffle = False, sampler = trainDataSampler, pin_memory=True)
    testDataLoader = DataLoader(test_data, batch_size = batchSize, shuffle = False, sampler = testDataSampler, pin_memory=True)

    return trainDataLoader, testDataLoader


def create_model(rank):
    
    device = torch.device(f'cuda:{rank}')
    model = UNet()
    model = model.to(device)
    model = DDP(model, device_ids=[rank], output_device=rank)
    opt = torch.optim.Adam(model.parameters())
    loss_func = nn.CrossEntropyLoss()
    return model, opt, loss_func

# def main(worldSize, rank, numEpochs, batchSize):

#     device = torch.device(f'cuda:{rank}')
#     model = UNet().to(device)
#     #ONLY DO IF YOU ARE USING MULTIPLE GPUs!
#     model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
#     # print('memory used: ' + str(torch.cuda.memory_allocated()) + '/' + str(torch.cuda.max_memory_allocated()))

#     #optimizer
#     opt = torch.optim.Adam(model.parameters())

#     #binary cross entropy loss function
#     # loss_func = nn.BCELoss()

#     #cross entropy loss function
#     loss_func = nn.CrossEntropyLoss().to(device)

#     #load data
#     # SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)
#     # sample = SMCCars[2193]
#     #number of epochs & batch size
#     numEpochs = 25
#     batchSize = 1

#     kwargs = {'num_workers': 1, 'pin_memory': True} if device == torch.device('cuda') else {}

#     #create random 80/20 split for train/test

#     #in case you want to load in model, there is a better way to do this (i.e., save train and test data locallys)
#     # torch.manual_seed(99)

#     # trainDataLoader = DataLoader(train_data, batch_size = batchSize, shuffle = True, **kwargs)
#     # testDataLoader = DataLoader(test_data, batch_size = batchSize, shuffle = True, **kwargs)

#     return model, batchSize, numEpochs

if __name__ == "__main__":

    print('program started')

    #load data
    rootdir = '/raid/gkroiz1/SMC21_GM_AV'



    #check for GPUs or CPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    n_gpus = torch.cuda.device_count()


    print('number of cuda devices: ' + str(n_gpus))

    #########################################################
    # world_size = n_gpus                                      #
    # os.environ['MASTER_ADDR'] = '10.57.23.164'              #
    # os.environ['MASTER_PORT'] = '8888'                      #
    # mp.spawn(training, nprocs=n_gpus)         #
    #########################################################

    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()

    numEpochs = 25
    batchSize = 5

    rank = args.local_rank
    print('rank: ' + str(rank))
    worldSize = torch.cuda.device_count()

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL,
                                         init_method='env://')

    trainDataLoader, testDataLoader = create_data_loaders(rank, worldSize, batchSize)
    if rank == 0:
        print('after creating dataLoaders: ')
        GPUtil.showUtilization()

    model, opt, loss_func = create_model(rank)
    if rank == 0:
        print('after creating model: ')
        GPUtil.showUtilization()
    #call training function
    training(rank, numEpochs, trainDataLoader, model, opt, loss_func)

    #load previously trained model
    # model = UNet()
    #ONLY DO IF YOU ARE USING MULTIPLE GPUs!
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # model.load_state_dict(torch.load('model.pt'))
    # model.to(device)

    #call testing function
    predictions(testDataLoader, model, batchSize)








