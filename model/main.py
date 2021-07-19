import matplotlib.pyplot as plt
import numpy as np
import os

# to make import from one dir up possible
import sys
sys.path.append("..") 

# import the custom Dataset object
import SMCCarsDataset


from torch.utils.data import DataLoader



#from model.py
from model import UNet

# from torchsummary import summary
import torch
import torch.nn as nn
# from pytorch_unet import UNet

import torch.optim as optim
from torch.optim import lr_scheduler
import copy
import time
from collections import defaultdict
# from trainer import Trainer
from tqdm import tqdm
from torchsummary import summary

#basic training function
def training(numEpochs, dataLoader, model, batchSize):
    print('in training')
    for epoch in range(numEpochs):
        for (batch_idx, batch) in enumerate(dataLoader):
            #define input and labels
            inputs = batch['image']
            labels = batch['segmentation']
            if batch_idx == 0 and epoch == 0:
                print(summary(model, input_size=(3, 1280, 720)))
            #output from model
            out = model(inputs)
            #calculate loss
            loss = loss_func(out, labels)
            # Backward
            opt.zero_grad()
            loss.backward()
            opt.step()

        print(f"\nEpoch: {epoch}/{numEpochs}, Loss: {loss}")
    torch.save(model.state_dict(), model)
    print("saved model")


if __name__ == "__main__":

    #load data
    rootdir = '../SMC21_GM_AV'
    # SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    #check for GPUs or CPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet(6).to(device)

    #optimizer
    opt = torch.optim.Adam(model.parameters())

    #binary cross entropy loss function
    loss_func = nn.BCELoss()

    #load data
    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    print(SMCCars.__len__())

    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())
    trainDataLoader = DataLoader(train_data, batch_size = 32, shuffle = True)
    testDataLoader = DataLoader(test_data, batch_size = 32, shuffle = True)

    #number of epochs
    numEpochs = 10
    batchSize = 16

    #call training function

    regularLoader = DataLoader(SMCCars, batch_size=batchSize, shuffle=True)
    training(numEpochs, trainDataLoader, model, batchSize)

