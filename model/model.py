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

#worth exploring dropout, normalization, more/less blocks, more frequent conv layers v maxpooling and concat


#conv2d -> ReLU -> Batch Normalization -> conv2d -> ReLU -> Batch Normalization -> MaxPool2D
def downBlock(in_channels, out_channels,kernel_size, stride = 1, padding = 0, normalization = True, dropout = 0.1):



    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size, stride, padding)
    )

#convTranspose2d -> ReLU -> BatchNorm2d -> Concatenate -> conv2d -> ReLU -> BatchNorm2d -> conv2d -> ReLU -> BatchNorm2d
def upBlock(in_channels, out_channels, kernel_size, stride = 1, padding = 0, normalization = True, dropout = 0.1):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
        nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding),
        nn.ReLU(),
        # nn.BatchNorm2d(out_channels),
    )
    return block

#worth implementing dropout, and normalization option
class UNet(nn.Module):

    def __init__(self, in_channels = 3, out_channels = 64, n_blocks = 4, start_filter_num = 64):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filter_num = start_filter_num

        self.dBlock1 = downBlock(3, 64, 2)
        self.dBlock2 = downBlock(64, 128, 2)
        self.dBlock3 = downBlock(128, 256, 2)
        self.dBlock4 = downBlock(256, 512, 2)

        
        #need to check dimensions of stuff
        self.uBlock1 = upBlock(512, 256, 2)
        self.uBlock2 = upBlock(256, 128, 2)
        self.uBlock3 = upBlock(128, 64, 2)
        # self.block8 = upBlock(64, )

        self.conv_last = nn.Conv2d(64, 2, 1)
    def forward(self, weights):
        weights = self.dBlock1(weights)
        weights = self.dBlock2(weights)
        weights = self.dBlock3(weights)
        weights = self.dBlock4(weights)
        weights = self.uBlock1(weights)
        weights = self.uBlock2(weights)
        weights = self.uBlock3(weights)
        out = self.conv_last(weights)
        return out
        #downBlock
        #downBlock
        #upBlock
        #upBlock
        #con2vD
