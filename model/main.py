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
# from pytorch_unet import UNet

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
# from tqdm import tqdm
from torchsummary import summary

#basic training function
def training(numEpochs, dataLoader, model, batchSize):
    print('in training')
    for epoch in range(numEpochs):
        for (batch_idx, batch) in enumerate(dataLoader):
            #define input and labels
            inputs = batch['image']
            labels = batch['segmentation']

            inputs = inputs.to(device)
            labels = labels.to(device)
            # inputs = inputs
            # labels = labels
            if batch_idx == 0 and epoch == 0:
                print(summary(model, input_size=(3, 1280, 720)))
            #output from model
            out = model(inputs)

            print(out.shape)
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
    print('program started')

    #load data
    rootdir = '/raid/gkroiz1/SMC21_GM_AV'
    # SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    #check for GPUs or CPUs
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = torch.device('cpu')
    print(device)
    model = UNet().to(device)
    # model = UNet()

    print(torch.cuda.device_count())
    # model = torch.nn.DataParallel(model, device_ids=list(range(torch.cuda.device_count())))
    print('memory used: ' + str(torch.cuda.memory_allocated()) + '/' + str(torch.cuda.max_memory_allocated()))

    #optimizer
    opt = torch.optim.Adam(model.parameters())

    #binary cross entropy loss function
    loss_func = nn.BCELoss()

    #load data
    SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

    print(SMCCars.__len__())
    # exit()

    # print(device == torch.device('cuda'))
    kwargs = {'num_workers': 1, 'pin_memory': True} if device == torch.device('cuda') else {}
    train_data, test_data = torch.utils.data.random_split(SMCCars, [int(np.ceil(SMCCars.__len__()*0.8)), int(np.floor(SMCCars.__len__()*0.2))], generator=torch.Generator())
    trainDataLoader = DataLoader(train_data, batch_size = 32, shuffle = True, **kwargs)
    testDataLoader = DataLoader(test_data, batch_size = 32, shuffle = True, **kwargs)

    #number of epochs
    numEpochs = 10
    batchSize = 1

    #call training function

    regularLoader = DataLoader(SMCCars, batch_size=batchSize, shuffle=True)
    training(numEpochs, trainDataLoader, model, batchSize)

