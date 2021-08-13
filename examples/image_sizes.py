"""
A script which counts the number of images of each size in the dataset
NO LONGER WORKS properly due to all images in Dataset now being returned 
as the same size.

Result:
{torch.Size([3, 720, 1280]): 4898, torch.Size([3, 1024, 2048]): 700}
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import torchvision
import torchvision.transforms as transforms

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

shapes = {}

for index, sample in enumerate(SMCCars):
    image = sample['image']
    dims = image.shape
    if dims in shapes:
        shapes[dims]+=1
    else:
        shapes[dims] = 1
    if index % 100 == 0:
        print(str(index) + ": " + str(shapes))
        
print(shapes)