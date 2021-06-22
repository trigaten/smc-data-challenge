"""
A code example that implements the SMCCardsDataset object, indexes it for 
a sample, and displays the sample with its segmentation map.
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import torchvision
import torchvision.transforms as transforms

# for visualization
import matplotlib.pyplot as plt
import numpy as np

# for creating a Pytorch Dataset object
from torch.utils.data import Dataset

import SMCCarsDataset

rootdir = 'SMC21_GM_AV'

SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

sample = SMCCars[0]

image = (sample['image'])
segmentation = (sample['segmentation'])

# show the image with its segmentation 
fig = plt.figure()

ax = fig.add_subplot(1, 2, 1)
imgplot = plt.imshow(image)
ax.set_title('Image')

ax = fig.add_subplot(1, 2, 2)
imgplot = plt.imshow(segmentation)
ax.set_title('Segmentation')

plt.show()