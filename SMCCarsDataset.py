
""" Dataset object for the car images adapted from 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html. This class 
extends Pytorch's Dataset class and makes the image data easily accesible.
Note: indexing is supported, but slicing is currently not.
    """

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
from torchvision.transforms import InterpolationMode

# for going through the folders
import os

# for creating a Pytorch Dataset object
from torch.utils.data import Dataset

# for returning images in the indexer
from torchvision.io import read_image

class SMCCarsDataset(Dataset):
    def __init__(self, root_dir, transform=None, returnCity=True,returnSynth=False):
        """
        Args:
            root_dir (string): Directory with all the images 
            and their segmentations (the SMC21_GM_AV folder)
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.returnCity = returnCity
        self.returnSynth = returnSynth
        # a list containing the path of every image
        self.image_list, self.seg_list = self.get_image_seg_list(root_dir)
        self.root_dir = root_dir

        self.transform = transform

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        """ The indexing method for this object. If you have an instance of this object you can 
        do instance[0] to get the first data sample, for example. 
        :return: a tuple of pytorch tensors-- the image and its segmentation map
        """
        if torch.is_tensor(idx):
            idx = idx.tolist()

        # convert path to image/seg to image/seg itself
        img_path = self.image_list[idx]
        seg_path = self.seg_list[idx]
        # check that image is not corrupt
        try:
            image = read_image(img_path)
        except Exception as e:
            raise Exception("Unable to read image at " + img_path + ". Verify that it is not corrupted")
        segmentation = read_image(seg_path)

        image = image.float()
        segmentation = segmentation.float()
        
        # remove the alpha channels
        # test if alpha channel exists-- not all pngs have alpha channel
        if (image.shape[0] == 4):
            image = torch.split(image, len(image)-1, 0)[0]
        if (segmentation.shape[0] == 4):
            segmentation = torch.split(segmentation, len(segmentation)-1, 0)[0]
        
        image, segmentation = self.resize(image, segmentation)

        sample = {'image': image, 'segmentation': segmentation}

        if self.transform:
            sample = self.transform(sample)

        return sample

    def resize(self, image, segmentation):
        if (image.shape[1] == 1024):
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1820))
            image = TF.crop(image, i, j, h, w)
            segmentation = TF.crop(segmentation, i, j, h, w)
            # NEAREST Interpolation so that segmap is logically interpolated
            resize = transforms.Resize((720, 1280), InterpolationMode.NEAREST)
            image = resize(image)
            segmentation = resize(segmentation)

        return (image, segmentation)

    def get_image_seg_list(self, dir):
        """ returns a tuple containing a list of all the image paths 
        and a list of all the segmentation image paths """
        images, segmentations = [], []
        for image_type_folder in os.listdir(dir):
            if image_type_folder == ".DS_Store":
                continue
            print(image_type_folder)

            if not self.returnCity:
                if image_type_folder == "Cityscapes":
                    continue
            if not self.returnSynth:
                if image_type_folder != "Cityscapes":
                    continue
            print("YES")
            # get the path of the folder (like Cityscapes of ClearNoon)
            image_type_path = os.path.join(dir, image_type_folder)

            # the path for images
            images_path = os.path.join(image_type_path, "images")
            # append all image paths to the list
            for image_file in os.listdir(images_path):
                # check for hidden .DS_STORE filess
                if image_file.find(".DS_Store") != -1:
                    continue
                images.append(os.path.join(images_path, image_file))

            # the path for the segmentations
            segmentations_path = os.path.join(image_type_path, "segmentations")
            # append all segmentation image paths to the list
            for segmentation_file in os.listdir(segmentations_path):
                segmentations.append(os.path.join(segmentations_path, segmentation_file))

        return (images, segmentations)

if __name__ == '__main__':
    rootdir = 'SMC21_GM_AV'
    # instantiate an instance of the Dataset object
    SMCCars = SMCCarsDataset(rootdir)
    sample = SMCCars[2193]
