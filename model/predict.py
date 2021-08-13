
""" Takes the model from main.py, and runs the test data through the model. 
This data is saved in the 'results' directory under 'gt' and 'pred 
"""

__author__ = "Gerson Kroiz"
__email__ = "gkroiz1@umbc.edu"


import matplotlib.pyplot as plt
import numpy as np
import os

# to make import from one dir up
import sys
sys.path.append("..") 
# import the custom Dataset object

import SMCCarsDataset
from SMCCarsDataset import onehot_to_rgb
import pickle


#import utilities
import utilities

import torch
import torch.nn as nn
import torch.optim as optim
import time
from collections import defaultdict

#import for multi-gpu training
from torch.distributed import Backend

from json import load as loadf
import argparse
from torchvision.io import write_png

#function for evaluating model
def predictions(dataLoader, model, batchSize, resultsdir):

    #convert model to eval mode
    model = model.eval()

    #create gt and pred directories
    if not os.path.isdir(resultsdir):
        os.mkdir(resultsdir)
    os.chdir(resultsdir)
    if not os.path.isdir('gt'):
        os.mkdir('gt')
    if not os.path.isdir('pred'):
        os.mkdir('pred')

    #do predictions one at a time
    for (batch_idx, batch) in enumerate(dataLoader):

        inputs = batch['image']
        labels = batch['segmentation']

        inputs = inputs.to(device, torch.float)

        #run input data through model
        predictions = model(inputs)

        #save predictions and ground truth in respective directories
        for index in range(batchSize):

            #convert the the dimensions of the predictions to dimensions of an image (RGB)
            tmp = onehot_to_rgb(predictions[index]).type(torch.uint8)

            label = labels[index]#.type(torch.uint8)

            #save images
            write_png(tmp, 'pred/image' + str(batch_idx * batchSize + index) + '.png')
            write_png(label, 'gt/image' + str(batch_idx * batchSize + index) + '.png')

if __name__ == "__main__":

    with open("main.json", 'r') as inFile:
        json_params = loadf(inFile)
    #load json file information
    rootdir = json_params["rootdir"]
    save_model_loc = json_params["save_model_loc"]
    resultsdir = json_params["resultsdir"]

    #load data
    rootdir = '/raid/gkroiz1/SMC21_GM_AV'
    resultsdir = 'default-results'
    # resultsdir = 'all-trans-results'
    # resultsdir = 'paper-results'
    # rootdir = '/raid/gkroiz1/SMC21_GM_AV_w_style'
    # save_model_loc = 'tmp.pt'

    save_model_loc = 'default-model.pt'
    # save_model_loc = 'all-trans-model.pt'

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

    #set batch size to 1 and load data
    batch_size = 1
    testDataLoader = utilities.create_data_loader(rootdir, rank, worldSize, batch_size, 'test')
    
    #define model, optimizer, and loss function
    model, opt, loss_func = utilities.create_model(rank)
    

    #load in trained model via command line argument
    model.load_state_dict(torch.load(save_model_loc, map_location=lambda storage, loc: storage))

    #call testing function
    predictions(testDataLoader, model, batch_size, resultsdir)
