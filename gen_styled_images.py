"""
Creates a copy of the dataset with each cityscapes image styled in a random non-cityscape way
and each non-cityscape image styled in a cityscape way.
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import os, random

from style_trans import style_transfer

from SMCCarsDataset import SMCCarsDataset

from torch.utils.data import Dataset, DataLoader

from torchvision.utils import save_image


rootdir = "SMC21_GM_AV"

save_path = rootdir + "/styled"

save_path_images = rootdir + "/styled/images"

save_path_segmentations = rootdir + "/styled/segmentations"




SMCCars_city = SMCCarsDataset(rootdir, returnCity=True, returnSynth=False)
SMCCars_synth = SMCCarsDataset(rootdir, returnCity=False, returnSynth=True)
# print(len(SMCCarsDataset))
dataloader_city = DataLoader(SMCCars_city, batch_size=1, shuffle = True)
dataloader_synth = DataLoader(SMCCars_synth, batch_size=1, shuffle = True)

os.mkdir(save_path)
os.mkdir(save_path_images)
os.mkdir(save_path_segmentations)

for index, sample in enumerate(SMCCars_city):
    image = sample['image']
    style_img = next(iter(dataloader_synth))
    styled_image = style_transfer(image, style_img)
    save_image(styled_image, save_path_images + '/c2simg' + str(index) + '.png')

for index, sample in enumerate(SMCCars_synth):
    image = sample['image']
    style_img = next(iter(dataloader_city))
    styled_image = style_transfer(image, style_img)
    save_image(styled_image, save_path_images + 's2c/img' + str(index) + '.png')



