"""
A code example that saves an image obtained by a dataloader
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.io import write_png

# for visualizatio
import matplotlib.pyplot as plt
import numpy as np

# for creating a Pytorch Dataset object
from torch.utils.data import Dataset, DataLoader

# to make import from one dir up possible
import sys
sys.path.append("..") 

# import the custom Dataset object
import SMCCarsDataset

rootdir = '../SMC21_GM_AV'

# instantiate an instance of the Dataset object
SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)
dataloader = DataLoader(SMCCars, batch_size=1, shuffle = True)
# print len of dataset-- how many samples

# get the first data sample
batch = next(iter(dataloader))
# get ordered list of images in batch
images = batch['image']
# get ordered list of segmentations in batch
segs = batch['segmentation']

# extract the image and its segmentation
image = images[0]

print(image)


write_png(image, "FFF.png")