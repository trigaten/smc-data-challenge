# File: main.py
# Author: Gerson Kroiz
# Description: uses model from model.py and
# creates train/test data, trains model, and creates
# gt and pred directories for evaluation

import matplotlib.pyplot as plt
import numpy as np
import os

# to make import from one dir up
import sys
sys.path.append("..") 
# import the custom Dataset object
import SMCCarsDataset


from torch.utils.data import DataLoader
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
# from trainer import Trainer
# from tqdm import tqdm
from torchsummary import summary

#basic training function
def training(numEpochs, dataLoader, model, batchSize):
    print('in training')
    for epoch in range(numEpochs):
        startEpoch = time.time()
        for (batch_idx, batch) in enumerate(dataLoader):

            #define input and labels
            inputs = batch['image']
            labels = batch['segmentation']
            
            #put inputs and labels on gpu if permitted
            inputs = inputs.to(device)
            labels = labels.to(device)

            #print model summary
            if batch_idx == 0 and epoch == 0:
                print(summary(model, input_size=(3, 1280, 720)))

            #output from model
            out = model(inputs)

            #calculate loss
            loss = loss_func(torch.sigmoid(out), torch.sigmoid(labels))

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

        




if __name__ == "__main__":
    print('program started')

    #load data
    rootdir = '/raid/gkroiz1/SMC21_GM_AV'

    # SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    #check for GPUs or CPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet().to(device)

    print('number of cuda devices: ' + str(torch.cuda.device_count()))

    #ONLY DO IF YOU ARE USING MULTIPLE GPUs!
    model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # print('memory used: ' + str(torch.cuda.memory_allocated()) + '/' + str(torch.cuda.max_memory_allocated()))

    #optimizer
    opt = torch.optim.Adam(model.parameters())

    #binary cross entropy loss function
    loss_func = nn.BCELoss()

    #load data
    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    #number of epochs & batch size
    numEpochs = 20
    batchSize = 8

    kwargs = {'num_workers': 1, 'pin_memory': True} if device == torch.device('cuda') else {}

    #create random 80/20 split for train/test
    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())
    trainDataLoader = DataLoader(train_data, batch_size = batchSize, shuffle = True, **kwargs)
    testDataLoader = DataLoader(test_data, batch_size = batchSize, shuffle = True, **kwargs)



    #call training function
    training(numEpochs, trainDataLoader, model, batchSize)

    #load previously trained model
    # model = UNet()
    #ONLY DO IF YOU ARE USING MULTIPLE GPUs!
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    # model.load_state_dict(torch.load('model.pt'))
    # model.to(device)

    #call testing function
    predictions(testDataLoader, model, batchSize)








