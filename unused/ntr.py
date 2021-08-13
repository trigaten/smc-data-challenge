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
against = [road_line, traffic_light, rider, motorcycle]
to = [road_line_to, traffic_light_to, rider_to, motorcycle_to]
# def do_comp(image, h, w):
#     for test_against, convert_to in zip(against, to):
#         if image[0][h][w] == test_against[0] and image[1][h][w] == test_against[1] and image[2][h][w] == test_against[2]:
#             image[0][h][w] = convert_to[0]
#             image[1][h][w] = convert_to[1]
#             image[2][h][w] = convert_to[2]

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
        size = segmentation.size()
        for h in range(size[1]):
            for w in range(size[2]):
                for test_against, convert_to in zip(against, to):
                    if segmentation[0][h][w] == test_against[0] and segmentation[1][h][w] == test_against[1] and segmentation[2][h][w] == test_against[2]:
                        segmentation[0][h][w] = convert_to[0]
                        segmentation[1][h][w] = convert_to[1]
                        segmentation[2][h][w] = convert_to[2]
        
                    
        write_png(segmentation, path)
        exit()
