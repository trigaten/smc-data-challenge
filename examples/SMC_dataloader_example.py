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

# to make import from one dir up possible
import sys
sys.path.append("..") 

import SMCCarsDataset

rootdir = '../SMC21_GM_AV'

SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir, traditional_transform = True, overlay_transform=True)
# print(len(SMCCarsDataset))
dataloader = DataLoader(SMCCars, batch_size=3, shuffle = True)

# get a batch
batch = next(iter(dataloader))
# get ordered list of images in batch
images = batch['image']
# get ordered list of segmentations in batch
segs = batch['segmentation']

# graph setup
loc = 1
fig = plt.figure(figsize=(8, 8))

# display first 3 images and segmentations of the batch
for index, image in enumerate(images):
    seg = segs[index]

    seg = seg.byte()
    image = image.byte() 

    image = image.permute(1, 2, 0)
    seg = seg.permute(1, 2, 0)

    ax = fig.add_subplot(3, 2, loc)
    imgplot = plt.imshow(image)
    ax.set_title('Image')

    loc+=1

    ax = fig.add_subplot(3, 2, loc)
    imgplot = plt.imshow(seg)
    ax.set_title('Segmentation')

    loc+=1
    if index == 2:
        break

plt.show()
