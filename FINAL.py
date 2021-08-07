""" a script that takes in a directory of images, runs semantic segmentation on 
each image, and then writes out the segmented image. """
# todo finish this when model fully trained
__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import sys
import torchvision.io as IO
import model from model.model.CnnUNet
from torchvision.utils import save_image

model = CnnUNet("USE_PRETRAINED_WEIGHTS_FLAG")

# assume folder name is passed as first argument (so is just after file path)
directory = sys.argv[1]

for index, image_file in enumerate(os.listdir(directory)):
    image = IO.read_image(img_path)
    pred_seg_map = model(torch.unsqueeze(image, 0))
    save_image(pred_seg_map, directory + "/seg" + str(index))



    