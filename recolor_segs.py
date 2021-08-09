import torch
import os
import torchvision.io as io
root_dir = "SMC21_GM_AV"
for image_type_folder in os.listdir(root_dir):
    if image_type_folder == ".DS_Store":
        continue
    # get the path of the folder (like Cityscapes of ClearNoon)
    image_type_path = os.path.join(root_dir, image_type_folder)

    # the path for the segmentations
    segmentations_path = os.path.join(image_type_path, "segmentations")
    # append all segmentation image paths to the list
    for segmentation_file in os.listdir(segmentations_path):
        path = os.path.join(segmentations_path, segmentation_file)
        
        segmentation = io.read_image(path)
        if segmentation.shape[0] == 4:
            segmentation = torch.split(segmentation, len(segmentation)-1, 0)[0]
            # print(segmentation.permute(1,  2, 0))
            for c in segmentation.permute(1,  2, 0):
                for r in c:
                    # print(r)
                    comp = torch.ByteTensor([0,0,0])
                    # print(comp)
                    if torch.equal(torch.ByteTensor([0,0,0]), r):
                        print(c)

            exit()