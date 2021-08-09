"""
Script which reads through all segmentation maps and converts some pixel values
to others as specified by challenge 
"""

__author__ = "Sander Schulhoff"
__email__ = "sanderschulhoff@gmail.com"

import os

from PIL import Image
import numpy as np

root_dir = "SMC21_GM_AV"

road_line = [157, 234, 50]
road_line_to = [128, 64, 128]
traffic_light = [250, 170, 30]
traffic_light_to = [220, 220, 0]
rider = [255, 0, 0]
rider_to = [220, 20, 60]
motorcycle = [0, 0, 230]
motorcycle_to = [119, 11, 32]

against = [road_line, traffic_light, rider, motorcycle]
to = [road_line_to, traffic_light_to, rider_to, motorcycle_to]

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
        segmentation = Image.open(path)
        segmentation = segmentation.convert('RGBA')
        data = np.array(segmentation)
        red, green, blue, a = data.T
        for test_against, convert_to in zip(against, to):
            r, g, b = test_against
            repl_areas = (red == r) & (green == g) & (blue == b)
            data[..., :-1][repl_areas.T] = convert_to
        im_out = Image.fromarray(data)
        im_out.save(path)
