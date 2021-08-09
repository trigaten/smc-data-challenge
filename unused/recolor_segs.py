import torch
import os
import torchvision.io as io
from torchvision.io import write_png
from torchvision.io import ImageReadMode

root_dir = "SMC21_GM_AV"

road_line = torch.ByteTensor([157, 234, 50])
road_line_to = torch.ByteTensor([128, 64, 128])
traffic_light = torch.ByteTensor([250, 170, 30])
traffic_light_to = torch.ByteTensor([220, 220, 0])
rider = torch.ByteTensor([255, 0, 0])
rider_to = torch.ByteTensor([220, 20, 60])
motorcycle = torch.ByteTensor([0, 0, 230])
motorcycle_to = torch.ByteTensor([119, 11, 32])

for image_type_folder in os.listdir(root_dir):
    print(image_type_folder)
    if image_type_folder == ".DS_Store":
        continue
    # get the path of the folder (like Cityscapes of ClearNoon)
    image_type_path = os.path.join(root_dir, image_type_folder)

    # the path for the segmentations
    segmentations_path = os.path.join(image_type_path, "segmentations")
    # append all segmentation image paths to the list
    for segmentation_file in os.listdir(segmentations_path):
        path = os.path.join(segmentations_path, segmentation_file)
        segmentation = io.read_image(path, ImageReadMode.RGB)
        # if segmentation.shape[0] == 4:
        #     segmentation = torch.split(segmentation, len(segmentation)-1, 0)[0]
        segmentation = segmentation.permute(1,  2, 0)
        for c in segmentation:
            for index, r in enumerate(c):
                if torch.equal(road_line, r):
                    c[index] = road_line_to
                elif torch.equal(traffic_light, r):
                    c[index] = traffic_light_to
                elif torch.equal(rider, r):
                    c[index] = rider_to
                elif torch.equal(motorcycle, r):
                    c[index] = motorcycle_to
        segmentation = segmentation.permute(2,  0, 1)
        write_png(segmentation, path)
