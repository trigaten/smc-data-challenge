""" a script that takes in a directory of images, runs semantic segmentation on 
each image, and then writes out the segmented image. """
# todo finish this when model fully trained
__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import sys
import torchvision.io as IO
import model from model.model.CnnUNet

model = CnnUNet("USE_PRETRAINED_WEIGHTS_FLAG")

# assume folder name is passed as first argument (so is just after file path)
dir = sys.argv[1]

for image_file in os.listdir(dir):
    image = IO.read_image(img_path)
    pred_seg_map = model(torch.unsqueeze(image, 0))
    



    