""" Dataset object for the car images adapted from 
https://pytorch.org/tutorials/beginner/data_loading_tutorial.html. This class
extends Pytorch's Dataset class and makes the image data easily accessible.
Note: indexing is supported, but slicing is currently not.
    """

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import torch
import random
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.io as IO
import os
from torch.utils.data import Dataset
import numpy as np


class SMCCarsDataset(Dataset):
    def __init__(self, root_dir, traditionalTransform=False, styleTransform=False, overlayTransform=False):
        """
        Args:
            root_dir (string): Directory with all the images 
                and their segmentations (the SMC21_GM_AV folder)
            traditionalTransform (callable, optional): Optional transform to be applied
                on a sample.
            styleTransform (callable, optional): Optional transform to be applied
                on a sample.
            overlayTransform (callable, optional): Optional transform to be applied
                on a sample.
        """
        # a list containing the path of every image
        self.image_list, self.seg_list = self.get_image_seg_list(root_dir)
        self.root_dir = root_dir

        self.traditionalTransform = traditionalTransform
        self.styleTransform = styleTransform
        self.overlayTransform = overlayTransform

        self.leastWidth = 1280
        self.leastHeight = 720

        self.resizeFactor = 0.75
        self.cropFactor = 0.5

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
            image = IO.read_image(img_path)
        except Exception as e:
            raise Exception("Unable to read image at " + img_path + ". Verify that it is not corrupted")
        segmentation = IO.read_image(seg_path)

        # image = image.float()
        # segmentation = segmentation.float()
        
        # remove the alpha channels
        # test if alpha channel exists-- not all pngs have alpha channel
        if image.shape[0] == 4:
            image = torch.split(image, len(image)-1, 0)[0]
        if segmentation.shape[0] == 4:
            segmentation = torch.split(segmentation, len(segmentation)-1, 0)[0]
        



        image, segmentation = self.resize(image, segmentation)

        if self.traditionalTransform:
            image, segmentation = self.transform(image, segmentation)

        segmentation = rgb_to_single(segmentation, color_dict)

        # segmentation = segmentation.float()
        image = image.float()

        sample = {'image': image, 'segmentation': segmentation}

        return sample

    def transform(self, img, seg):
        # To PIL
        image = TF.to_pil_image(img)
        segmentation = TF.to_pil_image(seg)

        """
        # Random Crop
        i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(
            int(self.cropFactor * self.leastHeight), int(self.cropFactor * self.leastWidth)))
        image = TF.crop(image, i, j, h, w)
        segmentation = TF.crop(segmentation, i, j, h, w)
        """

        # Random Flip
        if random.random() > 0.5:
            image = TF.hflip(image)
            segmentation = TF.hflip(segmentation)

        # Jitters colors
        jitter = transforms.ColorJitter(brightness=0.5, hue=0.5, contrast=0.5, saturation=0.5)
        image = jitter(image)

        # To Tensor
        image = TF.to_tensor(image)
        segmentation = TF.to_tensor(segmentation)

        # Normalize
        # image = TF.normalize(image, mean=self.meanImageColor, std=self.stdImageColor)

        return image, segmentation

    def resize(self, image, segmentation):
        if image.shape[1] == 1024:
            i, j, h, w = transforms.RandomCrop.get_params(image, output_size=(1024, 1820))
            image = TF.crop(image, i, j, h, w)
            segmentation = TF.crop(segmentation, i, j, h, w)
            # NEAREST Interpolation so that segmap is logically interpolated
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

#convert from a numpy array from (3, y, x) -> (15, y, x)
def rgb_to_onehot(segmentation, color_dict=color_dict):
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

def rgb_to_single(rgb_arr, color_dict):
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

#for converting from onehot to rgb: (15, y, x) -> (3, y, x)
def onehot_to_rgb(segmentation, color_dict=color_dict):
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

if __name__ == '__main__':
    rootdir = 'SMC21_GM_AV'
    # instantiate an instance of the Dataset object
    SMCCars = SMCCarsDataset(rootdir)
    sample = SMCCars[2193]
 
   # ax = fig.add_subplot(3, 2, 5)
    # imgplot = plt.imshow(new_im.permute(1, 2, 0))
    # ax.set_title('New Image')
    
    # ax = fig.add_subplot(3, 2, 6)
    # imgplot = plt.imshow(onehot_to_rgb(segmentation, color_dict).permute(1, 2, 0))
    # ax.set_title('New Seg')
    
    

    # plt.show()