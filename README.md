# Challenge 3 Submission for smc data challenge

## How to use this repository

### 1.1: Install conda environment dependencies
Here is a list of the dependencies for the conda environemnt. At the moment, this code only runs in a GPU environment with CUDA.
  - torchvision
  - pytorch
  - torchaudio
  - cudatoolkit
  - matplotlib
  - tqdm
  - requests
  - scikit-image
  - python version 3.8.10 or less
  - opencv

### 1.2: Remove corrupt images from the dataset
The following images are corrupt:
```SMC21_GM_AV/CloudyNoon/images/00000086.png```
```SMC21_GM_AV/HardRainNoon/images/00000029.png```

Remove them and their segmentations by running the Remove_Images.sh bash script:
1) First chmod it:
`chmod +x Remove_Images.sh`
2) Then run 
`./Remove_Images.sh`

### 1.3: Remove classifications with more than one RGB representation
Most segmaps are not entirely helpful due to project specs run fix_segs.py to fix this
```python fix_segs.py```

### 2.1: Train the model
If you want to use one of the provided models, go to 3.1

### 2.2: Apply style transforms
As of now, the style transforms take a long period of time. As such, this data augmentation techniques is done beforehand and the styled images are saved.
