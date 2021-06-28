"""
A code example that implements the SMCCardsDataset object, indexes it for 
a sample, and displays the sample with its segmentation map.
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
sample = SMCCars[2480]
shapes = set(())

index = 3631
while True:
    try:
        sample = SMCCars[index]
    except Exception as e:
        print(e)
    image = sample['image']
    dims = image.shape
    print(index)
    print(dims)
    index+=1

# for index, sample in enumerate(SMCCars[2400,:]):
#     image = sample['image']
#     dims = image.shape
#     shapes.add(dims)
#     if index % 100 == 0:
#         print(str(index) + ": " + str(shapes))
        
# print(shapes)