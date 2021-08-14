
""" 
Defines training and testing data, defines model, and trains model on the data.
"""

__author__ = "Gerson Kroiz"
__email__ = "gkroiz1@umbc.edu"

import matplotlib.pyplot as plt
import os

#import utilities
import utilities

import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchsummary import summary

#import for multi-gpu training
from torch.distributed import Backend

import argparse
from json import load as loadf
import sys

#basic training function
def training(rank, num_epochs, batch_size, dataLoader, model, opt, loss_func, save_model_loc):
    """
    Desscription: Trains the model over the data based on the number of epochs provided 
    Args:
        rank (int): process ID for the GPU
        num_epochs (int): number of epochs the model will run through for training
        batch_size (int): size of each batch
        dataLoader (object): object that loads in data
        model (object): UNet model as defined in model.py
        opt (object): optimizer function
        loss_func (object): cross-entropy loss function
        saved_model_loc (string): location for where to save the model
    """

    model.train()

    #for loop for number of epochs
    for epoch in range(num_epochs):
        startEpoch = time.time()
        dataLoader.sampler.set_epoch(epoch)
        epochLoss = 0.0
        numBatches = 0

        #iterate through the data via the data
        for (batch_idx, batch) in enumerate(dataLoader):

            numBatches += 1

            #define input and labels
            inputs = batch['image']
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
            loss = loss_func(out, labels)

            # back propogate
            opt.zero_grad()
            loss.backward()

            epochLoss += loss.item() * batch_size
            opt.step()

        endEpoch = time.time()

        #save checkout every 5 epochs
        if rank == 0 and epoch % 5 == 0:
            torch.save(model.state_dict(), 'checkpoint.pt')

        #print epoch information
        if rank == 0:
            print('Epoch: '+str(epoch)+'/'+str(num_epochs)+', Loss: '+str(epochLoss)+', Avg Loss: '+str(epochLoss /(numBatches * batch_size))+', Time: '+str(endEpoch-startEpoch))

    #save final model
    if rank == 0:
        torch.save(model.state_dict(), save_model_loc)
        print("Model saved.")


if __name__ == "__main__":

    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)
    #load json file information
    rootdir = json_params["rootdir"]
    save_model_loc = json_params["saved_model_loc"]
    num_epochs = json_params["num_epochs"]
    batch_size = json_params["batch_size"]


    #check for GPUs
    device = torch.device("cuda" if torch.cuda.is_available() else sys.exit('Cuda is required with current version'))
    n_gpus = torch.cuda.device_count()



    #get number of ranks and local rank
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int)
    args = parser.parse_args()
    rank = args.local_rank

    if rank == 0:
        print('number of cuda devices: ' + str(n_gpus))

    #for DDP
    worldSize = torch.cuda.device_count()
    torch.cuda.set_device(rank)
    torch.distributed.init_process_group(backend=Backend.NCCL, init_method='env://')

    #use both transforms for data loader but compress segmentations for loss function
    traditional_transform = True
    overlay_transform = True
    return_rgb = False

    #define data loaders
    trainDataLoader = utilities.create_data_loader(rootdir, rank, worldSize, batch_size, traditional_transform, overlay_transform, return_rgb, 'train')


    #define model, optimizer, and loss function
    model, opt, loss_func = utilities.create_model(rank)

    #if you load in a model
    checkpoint = json_params["checkpoint_loc"]
    model.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))


    #call training function
    training(rank, num_epochs, batch_size, trainDataLoader, model, opt, loss_func, save_model_loc)








