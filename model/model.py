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
def downBlock(in_channels, out_channels,kernel_size, stride, padding, normalization = True, dropout = 0.1):

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
        nn.BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels, kernel_size, padding),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.MaxPool2d(kernel_size, stride, padding)
    )

#convTranspose2d -> ReLU -> BatchNorm2d -> Concatenate -> conv2d -> ReLU -> BatchNorm2d -> conv2d -> ReLU -> BatchNorm2d
def upBlock(in_channels, out_channels, normalization = True, dropout = 0.1):
    block = nn.Sequential(
        nn.ConvTranspose2d(in_channels, out_channels),
        nn.ReLU(),
        nn.BatchNorm2d(out_channels),
        nn.Upsample(),
        nn.Conv2d(out_channels, out_channels),
        nn.ReLU(),
        nn>BatchNorm2d(out_channels),
        nn.Conv2d(out_channels, out_channels),
        nn.ReLU(),
        nn>BatchNorm2d(out_channels),
    )
    return block

#worth implementing dropout, and normalization option
class CnnUNet(nn.Module):

    def __init__(self, in_channels, out_channels, n_blocks, start_filter_num):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_blocks = n_blocks
        self.start_filter_num = start_filter_num

        self.block1 = downBlock(3, 64)
        self.block2 = downBlock(64, 128)
        
        #need to check dimensions of stuff
        self.block3 = upBlock()
        self.block4 = upBlock()

        self.conv_last = nn.Conv2d(64, 3, 1)
    def build_model(self, weights):
        weights = self.block1
        weights = self.block2
        weights = self.block3
        weights = self.block4
        out = self.conv_last(weights)
        return out
        #downBlock
        #downBlock
        #upBlock
        #upBlock
        #con2vD


####NOT MY CODE

from collections import defaultdict
import torch.nn.functional as F
from loss import dice_loss

# def calc_loss(pred, target, metrics, bce_weight=0.5):
#     bce = F.binary_cross_entropy_with_logits(pred, target)
        
#     pred = F.sigmoid(pred)
#     dice = dice_loss(pred, target)
    
#     loss = bce * bce_weight + dice * (1 - bce_weight)
    
#     metrics['bce'] += bce.data.cpu().numpy() * target.size(0)
#     metrics['dice'] += dice.data.cpu().numpy() * target.size(0)
#     metrics['loss'] += loss.data.cpu().numpy() * target.size(0)
    
#     return loss

# def print_metrics(metrics, epoch_samples, phase):    
#     outputs = []
#     for k in metrics.keys():
#         outputs.append("{}: {:4f}".format(k, metrics[k] / epoch_samples))
        
#     print("{}: {}".format(phase, ", ".join(outputs)))    

# def train_model(model, optimizer, scheduler, num_epochs=25):
#     best_model_wts = copy.deepcopy(model.state_dict())
#     best_loss = 1e10

#     for epoch in range(num_epochs):
#         print('Epoch {}/{}'.format(epoch, num_epochs - 1))
#         print('-' * 10)
        
#         since = time.time()

#         # Each epoch has a training and validation phase
#         for phase in ['train', 'val']:
#             if phase == 'train':
#                 scheduler.step()
#                 for param_group in optimizer.param_groups:
#                     print("LR", param_group['lr'])
                    
#                 model.train()  # Set model to training mode
#             else:
#                 model.eval()   # Set model to evaluate mode

#             metrics = defaultdict(float)
#             epoch_samples = 0
            
#             for inputs, labels in dataloaders[phase]:
#                 inputs = inputs.to(device)
#                 labels = labels.to(device)             

#                 # zero the parameter gradients
#                 optimizer.zero_grad()

#                 # forward
#                 # track history if only in train
#                 with torch.set_grad_enabled(phase == 'train'):
#                     outputs = model(inputs)
#                     loss = calc_loss(outputs, labels, metrics)

#                     # backward + optimize only if in training phase
#                     if phase == 'train':
#                         loss.backward()
#                         optimizer.step()

#                 # statistics
#                 epoch_samples += inputs.size(0)

#             print_metrics(metrics, epoch_samples, phase)
#             epoch_loss = metrics['loss'] / epoch_samples

#             # deep copy the model
#             if phase == 'val' and epoch_loss < best_loss:
#                 print("saving best model")
#                 best_loss = epoch_loss
#                 best_model_wts = copy.deepcopy(model.state_dict())

#         time_elapsed = time.time() - since
#         print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
#     print('Best val loss: {:4f}'.format(best_loss))

#     # load best model weights
#     model.load_state_dict(best_model_wts)
#     return model