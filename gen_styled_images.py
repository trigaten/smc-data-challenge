"""
Creates a copy of the dataset with each cityscapes image styled in a random non-cityscape way
and each non-cityscape image styled in a cityscape way. Takes in command line argument for where
the root directory is located
"""

__authors__ = ["Sander Schulhoff", "Gerson Kroiz"]
__email__ = "sanderschulhoff@gmail.com"

import os, random

from style_trans import style_transfer

from SMCCarsDataset import SMCCarsDataset

from torch.utils.data import Dataset, DataLoader

from torchvision.utils import save_image


#for image loader
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import InterpolationMode

import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
import torchvision.models as models

import copy

#for style transform
import os, random

import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

import sys

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


loader = transforms.Compose([
    transforms.ToTensor()])  # transform it into a torch tensor

def image_loader(image_loc, crop_params = {}):
    '''
    Description: reads in image so it is compatible with style transform.
    Args:
        image_loc (string): location of image
        crop_params (dict): contains i, j, h, and w for cropping. 
        This is so incase you want to apply the same random crop to two images
    '''
    image = Image.open(image_loc).convert('RGB')
    image = loader(image)

    if (image.shape[1] == 1024):
        if len(crop_params) != 0:
            i = crop_params['i']
            j = crop_params['j']
            h = crop_params['h']
            w = crop_params['w']
        else:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1820))
        print('i:' + str(i) + ', j: ' + str(j) + ', h: ' + str(h) + ', w: ' + str(w))
        crop_params = {'i': i, 'j': j, 'h': h, 'w': w}
        image = TF.crop(image, i, j, h, w)
        # NEAREST Interpolation so that segmap is logically interpolated
        resize = transforms.Resize((720, 1280), InterpolationMode.NEAREST)
        image = resize(image)


    return image, crop_params


rootdir = sys.argv[1]

#create paths based on rootdir
save_path = rootdir + "/styled"
save_path_images = save_path + "/images"
save_path_segmentations = save_path + "/segmentations"

#create dataset objects and then the data loaders
SMCCars_city = SMCCarsDataset(rootdir, returnCity=True, returnSynth=False)
SMCCars_synth = SMCCarsDataset(rootdir, returnCity=False, returnSynth=True)

dataloader_city = DataLoader(SMCCars_city, batch_size=1, shuffle = True)
dataloader_synth = DataLoader(SMCCars_synth, batch_size=1, shuffle = True)


#create directory structure
if not os.path.isdir(save_path):
    os.mkdir(save_path)
if not os.path.isdir(save_path_images):
    os.mkdir(save_path_images)
if not os.path.isdir(save_path_segmentations):
    os.mkdir(save_path_segmentations)

#load cnn
cnn = models.vgg19(pretrained=True).features.to(device).eval()


#iterate through all city images and style them to look like synthetic images
for index, sample in enumerate(SMCCars_city):
    image_loc = sample['image']
    seg_loc = sample['segmentation']

    print('image_loc: ' + str(image_loc) + 'seg_loc: ' + str(seg_loc))
    #load content, style, and segmentation images
    image, crop_params = image_loader(image_loc)

    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0).to(device, torch.float)
    seg, crop_params = image_loader(image_loc, crop_params)
    style_img_loc = next(iter(dataloader_synth))['image'][0]
    style_img = image_loader(style_img_loc)

    #apply style transform
    styled_image = style_transfer(cnn, image, style_img)


    #save the style transform and the corresponding segmentation
    #the segmentation does not change
    save_image(styled_image, save_path_images + '/c2simg' + str(index) + '.png')
    save_image(seg, save_path_segmentations + '/c2simg' + str(index) + '.png')


#iterate through all synthetic images and style them to look like city images
for index, sample in enumerate(SMCCars_synth):
    image_loc = sample['image']
    seg_loc = sample['segmentation']

    #load content, style, and segmentation images
    image, crop_params = image_loader(image_loc)

    # fake batch dimension required to fit network's input dimensions
    image = image.unsqueeze(0).to(device, torch.float)
    seg, crop_params = image_loader(image_loc, crop_params)
    style_img_loc = next(iter(dataloader_city))['image'][0]
    style_img = image_loader(style_img_loc)
    style_img = style_img.unsqueeze(0).to(device, torch.float)

    #apply style transform
    styled_image = style_transfer(cnn, image, style_img)

    #save the style transform and the corresponding segmentation
    #the segmentation does not change
    save_image(styled_image, save_path_images + '/s2cimg' + str(index) + '.png')
    save_image(seg, save_path_segmentations + '/s2cimg' + str(index) + '.png')




