"""
A code example that implements the SMCCardsDataset object, indexes it for 
a sample, and displays the sample with its segmentation map.
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
from torch.utils.data import Dataset

# to make import from one dir up possible
import sys
sys.path.append("..") 

# import the custom Dataset object
import SMCCarsDataset

rootdir = '../SMC21_GM_AV'

# instantiate an instance of the Dataset object
SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

# print len of dataset-- how many samples
print(len(SMCCars))

# get the first data sample
sample = SMCCars[0]

# extract the image and its segmentation
image = (sample['image'])
segmentation = (sample['segmentation'])

print(image)
print(segmentation.shape)


write_png(image, "FFF.png")