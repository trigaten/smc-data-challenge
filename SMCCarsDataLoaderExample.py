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
dataloader = DataLoader(SMCCars, batch_size=64, shuffle = True)

train_features, train_labels = next(iter(dataloader))
# print(f"Feature batch shape: {train_features.size()}")
# print(f"Labels batch shape: {train_labels.size()}")
# img = train_features[0].squeeze()
# label = train_labels[0]
# plt.imshow(img, cmap="gray")
# plt.show()
# print(f"Label: {label}")