""" Dataset object for the car images adapted from 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html. This class 
extends Pytorch's Dataset class and makes the image data easily accessible.
To use styled dataset just put the folder with styled images in your root_dir 
that you pass to this object """

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"
__credits__ = ["Sander Schulhoff", "Joshua Anantharaj", "Gerson Kroiz",
                    "Nick Drake"]

import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.io as IO
import os
from torch.utils.data import Dataset
import numpy as np
from matplotlib import pyplot as plt

color_dict = {0: (70, 70, 70), # Building
              1: (190, 153, 153), # Fence
              2: (153, 153, 153), # Pole
			  3: (244, 35, 232), # Sidewalk
			  4: (107, 142, 35), # Vegetation
			  5: (102, 102, 156), # Wall 
			  6: (128, 64, 128), # Road / road line
			  7: (220, 220, 0), # Traffic light / sign
			  8: (220, 20, 60), # Person / rider
			  9: (0, 0, 142), # Car
			  10: (0, 0, 70), # Truck
			  11: (0, 60, 100), # Bus
			  12: (0, 80, 100), # Train
			  13: (119, 11, 32), # Motorcycle / Bicycle
			  14: (0, 0, 0), # Anything else
			  }

# what object types can be Copy+Pasted in overlay transform
overlay_indices = [0, 2, 5, 7, 8, 9, 10, 11, 12, 13]

class SMCCarsDataset(Dataset):
    def __init__(self, root_dir, traditional_transform=False, 
        overlay_transform=False, return_rgb=False):
        """
        Args:
            root_dir (string): Directory with all the images 
                and their segmentations (the SMC21_GM_AV folder)
            traditional_transform (callable, optional): Specifies whether to
                apply traditional transforms
            overlay_transform (callable, optional): Specifies whether to 
                apply overlay AKA Copy+Paste transform
        """
        self.root_dir = root_dir

        self.traditional_transform = traditional_transform
        self.overlay_transform = overlay_transform

        self.return_rgb = return_rgb

        # lists containing paths of images and their segmaps
        self.image_list, self.seg_list = self.get_image_seg_list(root_dir)


    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx, for_overlay=False):
        """ The indexing method for this object. If you have an instance of this 
        object you can do instance[10] to get the 10th data sample, for example. 
        :return: a tuple of pytorch tensors-- an image and its segmentation map
        """

        # extract image, seg paths
        img_path = self.image_list[idx]
        seg_path = self.seg_list[idx]

        # handle corrupt images (these occurred in given dataset)
        try:
            image = IO.read_image(img_path)
        except Exception:
            raise Exception("Unable to read image at " + img_path + ". Verify that it is not corrupted.")
        
        # read seg map
        segmentation = IO.read_image(seg_path)

        # remove the alpha channels
        # test if alpha channel exists-- not all pngs have alpha channel
        if image.shape[0] == 4:
            image = torch.split(image, len(image)-1, 0)[0]
        if segmentation.shape[0] == 4:
            segmentation = torch.split(segmentation, len(segmentation)-1, 0)[0]
        
        # apply resize transform
        image, segmentation = self.resize(image, segmentation)
        
        # optionally apply traditional transforms
        if self.traditional_transform:
            image, segmentation = self.transform(image, segmentation)

        # optionally apply Copy+Paste transform
        if self.overlay_transform and not for_overlay:
            image, segmentation = self.overlay(image, segmentation)

        # optionally convert seg map to 15 channel 1-hot encoding 
        if not for_overlay and not self.return_rgb:
            segmentation = rgb_to_single(segmentation, color_dict)

        sample = {'image': image, 'segmentation': segmentation}

        return sample

    def transform(self, img, seg):
        # Random Flip
        if random.random() > 0.5:
            img = TF.hflip(img)
            seg = TF.hflip(seg)

        # Jitters colors
        jitter = transforms.ColorJitter(brightness=0.1, hue=0.1, contrast=0.1, saturation=0.1)
        img = jitter(img)

        return img, seg
    
    def overlay(self, img, seg):
        # 50% chance of applying the transform
        if random.random() > 0.5:
            return img, seg

        # convert to numpy
        new_sample_seg = seg
        new_sample_seg = new_sample_seg.detach().numpy()

        #randomly select new image to extract segmentation from
        #for_overlay=True prevents code from entering infinite loop
        random_content_sample = self.__getitem__(random.randint(0, len(self)-1), for_overlay=True)
        
        # convert seg to numpy
        random_content_seg = random_content_sample['segmentation']
        random_content_seg = random_content_seg.detach().numpy()

        # convert image to numpy
        random_content_img = random_content_sample['image']
        random_content_img = random_content_img.detach().numpy()

        # select index of what object type to copy
        index = random.choice(overlay_indices)

        # unpack color dict at the index
        r, g, b = color_dict[index]

        # unpack random seg into r, g, b channels
        red, green, blue = random_content_seg

        # form mask of areas to replace
        repl_areas = (red == r) & (green == g) & (blue == b)

        # perform replacement on seg map
        new_sample_seg[0][repl_areas], new_sample_seg[1][repl_areas], new_sample_seg[2][repl_areas] = [r, g, b]

        # convert to numpy
        new_image = img
        new_image = new_image.detach().numpy()
       
        # perform replacement on image
        new_image[0][repl_areas], new_image[1][repl_areas], new_image[2][repl_areas] = random_content_img[0][repl_areas], random_content_img[1][repl_areas], random_content_img[2][repl_areas]
        
        # convert to tensors
        new_sample_seg = torch.ByteTensor(new_sample_seg)
        new_image = torch.ByteTensor(new_image)
        
        return new_image, new_sample_seg
    
    def resize(self, image, segmentation):
        # Resize image if the image is larger than 1280x720
        if image.shape[1] == 1024:
            # Random crop to correct aspect ratio
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1820))
            image = TF.crop(image, i, j, h, w)
            segmentation = TF.crop(segmentation, i, j, h, w)
            # NEAREST Interpolation so that segmap is logically interpolated
            # Scale down to 1280x720
            resize = transforms.Resize((720, 1280), transforms.InterpolationMode.NEAREST)
            image = resize(image)
            segmentation = resize(segmentation)

        return image, segmentation

    def get_image_seg_list(self, dir):
        """ returns a tuple containing a list of all the image paths 
        and a list of all the segmentation image paths """
        images, segmentations = [], []
        for image_type_folder in os.listdir(dir):
            if image_type_folder == ".DS_Store":
                continue
            # get the path of the folder (like Cityscapes of ClearNoon)
            image_type_path = os.path.join(dir, image_type_folder)

            # the path for images
            images_path = os.path.join(image_type_path, "images")
            # append all image paths to the list
            for image_file in os.listdir(images_path):
                # check for hidden .DS_STORE files
                if image_file.find(".DS_Store") != -1:
                    continue
                images.append(os.path.join(images_path, image_file))

            # the path for the segmentations
            segmentations_path = os.path.join(image_type_path, "segmentations")
            # append all segmentation image paths to the list
            for segmentation_file in os.listdir(segmentations_path):
                segmentations.append(os.path.join(segmentations_path, segmentation_file))

        return (images, segmentations)

def rgb_to_onehot(segmentation, color_dict=color_dict):
    """convert from a numpy array from (3, y, x) -> (15, y, x)"""
    rgb_arr = segmentation.cpu().detach().numpy()
    rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    #print(num_classes, shape)
    arr = np.zeros( shape, dtype=np.int8 )
    #print(arr, arr.shape)
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    arr = np.transpose(arr, (2, 0, 1))
    arr = torch.from_numpy(arr)
    return arr

def onehot_to_rgb(segmentation, color_dict=color_dict):
    """for converting from onehot to rgb: (15, y, x) -> (3, y, x)"""
    onehot = segmentation.cpu().detach().numpy()
    onehot = np.transpose(onehot, (1, 2, 0))
    single_layer = np.argmax(onehot, axis=-1)
    output = np.zeros( onehot.shape[:2]+(3,) )
    for k in color_dict.keys():
        output[single_layer==k] = color_dict[k] 
    output = np.uint8(output)
    output = np.transpose(output, (2, 0, 1))
    output = torch.from_numpy(output)
    return output

def rgb_to_single(rgb_arr, color_dict):
    """convert from a numpy array from (3, y, x) -> (1, y, x)"""
    rgb_arr = rgb_arr.cpu().detach().numpy()
    rgb_arr = np.transpose(rgb_arr, (1, 2, 0))
    num_classes = len(color_dict)
    onehotshape = rgb_arr.shape[:2]+(num_classes,)
    singlelayershape = rgb_arr.shape[:2]+(1,)
    arr = np.zeros(singlelayershape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,0] += np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(onehotshape[:2]) * i
    arr = np.transpose(arr, (2, 0, 1))
    arr = torch.from_numpy(arr)
    return arr

if __name__ == '__main__':
    rootdir = 'SMC21_GM_AV'
    # instantiate an instance of the Dataset object
    SMCCars = SMCCarsDataset(rootdir)
    
    sample = SMCCars[20]
    
    print(sample)
        
