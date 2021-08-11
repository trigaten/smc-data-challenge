# smc-data-challenge

## The following images are corrupt

SMC21_GM_AV/CloudyNoon/images/00000086.png

SMC21_GM_AV/HardRainNoon/images/00000029.png

## Remove them and their segmentations by running the Remove_Images.sh bash script:

### First chmod it:

`chmod +x Remove_Images.sh`

### Then run 

`./Remove_Images.sh`

## Notes about running the model:
1) Run main.py
2) run evalulate.py for prediction results (python3 evalulate.py ./results)

## conda environment dependencies
  - torchaudio
  - torchvision
  - pytorch
  - matplotlib
  - scikit-image
  - pytorch-model-summary
