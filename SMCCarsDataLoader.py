"""
A code example that implements the SMCCardsDataset object with a DataLoader
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import torchvision
import torchvision.transforms as transforms


# for visualization
import matplotlib.pyplot as plt
import numpy as np

# for creating a Pytorch Dataset object and using it
from torch.utils.data import Dataset, DataLoader

import SMCCarsDataset

rootdir = 'SMC21_GM_AV'

SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)
# print(len(SMCCarsDataset))
dataloader = DataLoader(SMCCars, batch_size=4, shuffle = True, num_workers=0)

# for i_batch, sample_batched in enumerate(dataloader):
#     print(i_batch)