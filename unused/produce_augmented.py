from torchvision.utils import save_image
from dataset import SMCCarsDataset
import os

workingDir = os.getcwd() + "/"
augmentedDir = workingDir + 'Augmented/'

sets = ['Cityscapes/',
        'ClearNoon/',
        'CloudyNoon/',
        'CloudySunset/',
        'Default/',
        'HardRainNoon/',
        'MidRainSunset/',
        'SoftRainNoon/']


for currentSet in sets:
    root_dir = workingDir + currentSet
    dataset = SMCCarsDataset(root_dir=root_dir, traditionalTransform=True)

    count = 0
    print(currentSet)
    for sample in dataset:
        save_image(sample['image'], augmentedDir + "traditional/" + currentSet + '/images/img_' + str(count) + '.png')
        save_image(sample['segmentation'], augmentedDir + "traditional/" + currentSet + '/segmentations/img_' + str(count) + '.png')

        count += 1
        if count > 10:
            break
"""
for currentSet in sets:
    root_dir = workingDir + currentSet
    dataset = SMCCarsDataset(root_dir=root_dir, styleTransform=True)

    count = 0
    print(currentSet)
    for sample in dataset:
        save_image(sample['image'], augmentedDir + "style-transfer/" + currentSet + '/images/img_' + str(count) + '.png')
        save_image(sample['segmentation'], augmentedDir + "style-transfer/" + currentSet + '/segmentations/img_' + str(count) + '.png')

        count += 1
        if count > 10:
            break

for currentSet in sets:
    root_dir = workingDir + currentSet
    dataset = SMCCarsDataset(root_dir=root_dir, overlayTransform=True)

    count = 0
    print(currentSet)
    for sample in dataset:
        save_image(sample['image'], augmentedDir + "overlay/" + currentSet + '/images/img_' + str(count) + '.png')
        save_image(sample['segmentation'], augmentedDir + "overlay/" + currentSet + '/segmentations/img_' + str(count) + '.png')

        count += 1
        if count > 10:
            break
"""
