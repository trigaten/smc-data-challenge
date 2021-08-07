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

import numpy as np

# for going through the folders
import os

# for creating a Pytorch Dataset object
from torch.utils.data import Dataset

# for returning images in the indexer
from torchvision.io import read_image

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

#not needed rn			  
color_list = [(70, 70, 70),
			  (190, 153, 153),
			  (153, 153, 153),
			  (244, 35, 232),
			  (107, 142, 35),
			  (102, 102, 156),
			  (128, 64, 128),
			  (220, 220, 0),
			  (220, 20, 60),
			  (0, 0, 142),
			  (0, 0, 70),
			  (0, 60, 100),
			  (0, 80, 100),
			  (119, 11, 32),
			  ]


#convert from a numpy array from (y, x, 3) -> (y, x, 15)
def rgb_to_onehot(rgb_arr, color_dict):
    num_classes = len(color_dict)
    shape = rgb_arr.shape[:2]+(num_classes,)
    #print(num_classes, shape)
    arr = np.zeros( shape, dtype=np.int8 )
    #print(arr, arr.shape)
    for i, cls in enumerate(color_dict):
        arr[:,:,i] = np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(shape[:2])
    return arr

def rgb_to_single(rgb_arr, color_dict):
    num_classes = len(color_dict)
    onehotshape = rgb_arr.shape[:2]+(num_classes,)
    singlelayershape = rgb_arr.shape[:2]+(1,)
    arr = np.zeros(singlelayershape, dtype=np.int8 )
    for i, cls in enumerate(color_dict):
        arr[:,:,0] += np.all(rgb_arr.reshape( (-1,3) ) == color_dict[i], axis=1).reshape(onehotshape[:2]) * i
    return arr
#for converting from onehot to rgb; not needed rn
#def onehot_to_rgb(onehot, color_dict):
#    single_layer = np.argmax(onehot, axis=-1)
#    output = np.zeros( onehot.shape[:2]+(3,) )
#    for k in color_dict.keys():
#        output[single_layer==k] = color_dict[k]
#    return np.uint8(output)

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
        
        #tensor to numpy array for use in function
        seg_arr = segmentation.cpu().detach().numpy()
        #transpose array for function
        seg_arr = np.transpose(seg_arr, (1, 2, 0))
        #convert
        seg_arr = rgb_to_single(seg_arr, color_dict)
        
        seg_arr = np.transpose(seg_arr, (2, 0, 1))
        #numpy array to tensor
        segmentation = torch.from_numpy(seg_arr)

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
    rootdir = 'smcdc_c3'
    # instantiate an instance of the Dataset object
    SMCCars = SMCCarsDataset(rootdir)
    sample = SMCCars[2193]
    #print(sample)
