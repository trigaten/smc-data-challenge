
import matplotlib.pyplot as plt
import torch

# to make import from one dir up possible
import sys
sys.path.append("..") 

import random

# import the custom Dataset object
import SMCCarsDataset
from SMCCarsDataset import rgb_to_onehot 
from SMCCarsDataset import onehot_to_rgb 

rootdir = '../SMC21_GM_AV'

# instantiate an instance of the Dataset object
SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

sample = SMCCars[0]

new_sample_seg = torch.clone(sample['segmentation'])
new_sample_seg = rgb_to_onehot(new_sample_seg)

# get a random sample
random_content_sample = SMCCars[20]#[random.randint(0, len(SMCCars)-1)]

random_content_seg = random_content_sample['segmentation']
random_content_seg = rgb_to_onehot(random_content_seg)

# pick a random channel index to swap
index = 9 #random.randint(0, new_sample_seg.size()[0]-1)

# add the channel to sample
new_sample_seg[index] = torch.max(new_sample_seg[index], random_content_seg[index]) 

new_image = torch.where(random_content_seg[index] > 0, random_content_sample['image'], sample['image'])

fig = plt.figure(figsize=(8, 8))
# exit()
ax = fig.add_subplot(3, 2, 1)
imgplot = plt.imshow(sample['image'].permute(1, 2, 0))
ax.set_title('First image')

ax = fig.add_subplot(3, 2, 2)
imgplot = plt.imshow(sample['segmentation'].permute(1, 2, 0))
ax.set_title('First seg')

ax = fig.add_subplot(3, 2, 3)
imgplot = plt.imshow(random_content_sample['image'].permute(1, 2, 0))
ax.set_title('Second image')

ax = fig.add_subplot(3, 2, 4)
imgplot = plt.imshow(random_content_sample['segmentation'].permute(1, 2, 0))
ax.set_title('Second seg')

ax = fig.add_subplot(3, 2, 5)
imgplot = plt.imshow(new_image.permute(1, 2, 0))
ax.set_title('New Image')

ax = fig.add_subplot(3, 2, 6)

imgplot = plt.imshow(onehot_to_rgb(new_sample_seg).permute(1, 2, 0))
ax.set_title('New Seg')

plt.show()
