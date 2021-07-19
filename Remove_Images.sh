#!/bin/bash
# This script removes known corrupted images and their segmentations from the dataset
# @author Sander Schulhoff
# @email sanderschulhoff@gmail.com

# remove corrupt images
rm SMC21_GM_AV/CloudyNoon/images/00000086.png
rm SMC21_GM_AV/HardRainNoon/images/00000029.png

# remove their segmentations
rm SMC21_GM_AV/CloudyNoon/segmentations/00000086.png
rm SMC21_GM_AV/HardRainNoon/segmentations/00000029.png