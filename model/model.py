# File: model.py
# Author: Gerson Kroiz
# Description: UNET implementation 
# in pytorch

import os
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


#block structure based on up and down blocks from here:
#https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862
#https://divamgupta.com/image-segmentation/2019/06/06/deep-learning-semantic-segmentation-keras.html
#design
#worth exploring dropout, normalization, more/less blocks, more frequent conv layers v maxpooling and concat


#conv2d -> ReLU -> Batch Normalization -> conv2d -> ReLU -> Batch Normalization -> MaxPool2D
def block(in_channels, out_channels,kernel_size, stride = 1, padding = 1, normalization = True, dropout = 0.1):



    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
    )

#worth implementing dropout, and normalization option
class UNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 64, kernel_size = 3):
        super().__init__()
        # self.in_channels = in_channels
        # self.out_channels = out_channels
        # self.n_blocks = n_blocks
        # self.start_filter_num = start_filter_num

        self.dBlock1 = block(in_channels, out_channels, kernel_size)

        new_in_channel = out_channels
        new_out_channel = new_in_channel * 2
        self.dBlock2 = block(new_in_channel, new_out_channel, kernel_size)

        new_in_channel = new_out_channel
        new_out_channel = new_in_channel * 2
        self.dBlock3 = block(new_in_channel, new_out_channel, kernel_size)
        # self.dBlock4 = downBlock(256, 512, 3)

        
        #need to check dimensions of stuff
        # self.uBlock3 = block(512+256, 256, 3)

        # 128 + 256

        new_in_channel = int(new_out_channel/2) + new_out_channel
        new_out_channel = new_in_channel - new_out_channel
        self.uBlock2 = block(new_in_channel, new_out_channel, kernel_size)

        new_in_channel = int(new_out_channel/2) + new_out_channel
        new_out_channel = new_in_channel - new_out_channel
        self.uBlock1 = block(new_in_channel, new_out_channel, kernel_size)
        # self.block8 = upBlock(64, )

        self.conv_last = nn.Conv2d(new_out_channel, kernel_size, 1)
    def forward(self, weights):


        down1 = self.dBlock1(weights)
        pool1 = nn.MaxPool2d(2)(down1)

        down2 = self.dBlock2(pool1)
        pool2 = nn.MaxPool2d(2)(down2)

        down3 = self.dBlock3(pool2)

        weights = nn.Upsample(scale_factor=2)(down3)
        weights = torch.cat([weights, down2], dim=1)
        up2 = self.uBlock2(weights)

        weights = nn.Upsample(scale_factor=2)(up2)
        weights = torch.cat([weights, down1], dim=1)
        up1 = self.uBlock1(weights)


        out = self.conv_last(up1)
        return out
  
