"""
A script which transfers part of an image onto another image
NO LONGER WORKS due to Dataset object changes
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

# to make import from one dir up possible
import sys
sys.path.append("..") 

# import the custom Dataset object
import SMCCarsDataset

rootdir = '../SMC21_GM_AV'
# Number for the feature of image to transfer
feat_val = 70
# instantiate an instance of the Dataset object
SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

# get first sample
sample_0 = SMCCars[6]
# get its seg
sample_0_seg = torch.clone(sample_0['segmentation'])
# set non-feature values to 0
sample_0_seg[sample_0_seg!=feat_val] = 0
# sum tensor across channels
sample_0_seg_mask = torch.sum(sample_0_seg, 0)
# get image
sample_0_image = torch.clone(sample_0['image'])

sample_1 = SMCCars[200]
sample_1_seg = torch.clone(sample_1['segmentation'])
sample_1_image = torch.clone(sample_1['image'])

new_image = torch.where(sample_0_seg_mask > 0, sample_1['image'], sample_0['image'])

new_seg = torch.where(sample_0_seg_mask > 0, sample_1['segmentation'], sample_0['segmentation'])

# # show the image with its segmentation 
fig = plt.figure(figsize=(8, 8))


ax = fig.add_subplot(3, 2, 1)
imgplot = plt.imshow(sample_0['image'].permute(1, 2, 0))
ax.set_title('First image')

ax = fig.add_subplot(3, 2, 2)
imgplot = plt.imshow(sample_0['segmentation'].permute(1, 2, 0))
ax.set_title('First seg')

ax = fig.add_subplot(3, 2, 3)
imgplot = plt.imshow(sample_1['image'].permute(1, 2, 0))
ax.set_title('Second image')

ax = fig.add_subplot(3, 2, 4)
imgplot = plt.imshow(sample_1['segmentation'].permute(1, 2, 0))
ax.set_title('Second seg')

ax = fig.add_subplot(3, 2, 5)
imgplot = plt.imshow(new_image.permute(1, 2, 0))
ax.set_title('New Image')

ax = fig.add_subplot(3, 2, 6)
imgplot = plt.imshow(new_seg.permute(1, 2, 0))
ax.set_title('New Seg')



plt.show()


