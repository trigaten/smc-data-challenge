import os
import torch
from torch import nn
from torch.nn.modules.activation import ReLU
from torch.nn.modules.batchnorm import BatchNorm2d
from torch.nn.modules.conv import Conv2d
from torch.nn.modules.pooling import MaxPool2d
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

# device = 'cuda' if torch.cuda.is_available() else 'cpu'


#block structure based on up and down blocks from here:
#https://towardsdatascience.com/creating-and-training-a-u-net-model-with-pytorch-for-2d-3d-semantic-segmentation-model-building-6ab09d6a0862

def conv2d_block():

#conv2d -> ReLU -> Batch Normalization -> conv2d -> ReLU -> Batch Normalization -> MaxPool2D
def downBlock(in_channels, out_channels,kernel_size, padding):

    # # 4 in channels: R,G,B,alpha
    # in_channels = 4
    # out_channels = 16

    # #3x3 or 5x5 is common for kernel size, idk y
    # kernel_size = 3

    # #padding
    # padding = 0

    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, padding),
        nn.ReLU(),
        nn.BatchNorm2d(),
        nn.Conv2d(in_channels, out_channels, kernel_size, padding),
        nn.ReLU(),
        nn.BatchNorm2d(),
        nn.MaxPool2d(kernel_size, size)
    )

#convTranspose2d -> ReLU -> BatchNorm2d -> Concatenate -> conv2d -> ReLU -> BatchNorm2d -> conv2d -> ReLU -> BatchNorm2d
def upBlock(in_channels, out_channels):
    return nn.Sequential(
        nn.ConvTranspose2d(),
        nn.ReLU(),
        nn.BatchNorm2d(),
        nn.Upsample(),
        nn.Conv2d(),
        nn.ReLU(),
        nn>BatchNorm2d(),
        nn.Conv2d(),
        nn.ReLU(),
        nn>BatchNorm2d(),
    )

class UNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, start_filter_num):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filter_num = start_filter_num

    def build_model(self):

        #downBlock
        #downBlock
        #upBlock
        #upBlock
        #con2vD