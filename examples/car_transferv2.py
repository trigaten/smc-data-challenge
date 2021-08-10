
import matplotlib.pyplot as plt
import torch
import numpy as np
# to make import from one dir up possible
import sys
sys.path.append("..") 

import random

# import the custom Dataset object
import SMCCarsDataset
from SMCCarsDataset import rgb_to_onehot 
from SMCCarsDataset import onehot_to_rgb 

color_dict = {0: (70, 70, 70), # Building
              1: (190, 153, 153), # Fence
              2: (153, 153, 153), # Pole
			  3: (244, 35, 232), # Sidewalk
			  4: (107, 142, 35), # Vegetation
			  5: (102, 102, 156), # Wall 
			  6: (128, 64, 128), # Road / road line
			  7: (220, 220, 0), # Traffic light / sign
			  8: (220, 20, 60), # Person / rider
			  9: (0, 0, 142), # Car
			  10: (0, 0, 70), # Truck
			  11: (0, 60, 100), # Bus
			  12: (0, 80, 100), # Train
			  13: (119, 11, 32), # Motorcycle / Bicycle
			  14: (0, 0, 0), # Anything else
			  }

rootdir = '../SMC21_GM_AV'

# instantiate an instance of the Dataset object
SMCCars = SMCCarsDataset.SMCCarsDataset(rootdir)

sample = SMCCars[10]

new_sample_seg = torch.clone(sample['segmentation'])
new_sample_seg = new_sample_seg.detach().numpy()

# get a random sample
random_content_sample = SMCCars[20]#[random.randint(0, len(SMCCars)-1)]

random_content_seg = random_content_sample['segmentation']
random_content_seg = random_content_seg.detach().numpy()

random_content_img = random_content_sample['image']
random_content_img = random_content_img.detach().numpy()

# pick a random channel index to swap
index = 9 #random.randint(0, new_sample_seg.size()[0]-1)
# color_dict

# get colors to copy
r, g, b = color_dict[index]

red, green, blue = random_content_seg

repl_areas = (red == r) & (green == g) & (blue == b)

print(new_sample_seg)
# print(new_sample_seg[0].shape)
new_sample_seg[0][repl_areas], new_sample_seg[1][repl_areas], new_sample_seg[2][repl_areas] = [r, g, b]

new_image = torch.clone(sample['image'])
new_image = new_image.detach().numpy()

new_image[0][repl_areas], new_image[1][repl_areas], new_image[2][repl_areas] = random_content_img[0][repl_areas], random_content_img[1][repl_areas], random_content_img[2][repl_areas]


new_sample_seg = torch.ByteTensor(new_sample_seg)
new_image = torch.ByteTensor(new_image)
random_content_seg = torch.ByteTensor(random_content_seg)

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
