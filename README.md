# Challenge 3 submission for smc data challenge
Please follow the steps below to use this repository.
* Step 1 goes over what to install and how to preprocess the data before using the model.
* Step 2 goes over training your own models using the provided data augmentation techniques.
* Step 3 goes over creating predictions and runing evaluations.

Both Step 2 and 3 are located within the ```model``` directory.

## Step 1: 
### Step 1.1: Install conda environment dependencies
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

### Step 1.2: Remove corrupt images from the dataset
The following images are corrupt:
```SMC21_GM_AV/CloudyNoon/images/00000086.png```
```SMC21_GM_AV/HardRainNoon/images/00000029.png```

Remove them and their segmentations by running the Remove_Images.sh bash script:
1) First chmod it:
`chmod +x Remove_Images.sh`
2) Then run 
`./Remove_Images.sh`

### Step 1.3: Remove classifications with more than one RGB representation
Most segmaps are not entirely helpful due to the project specificiations. Run fix_segs.py to fix this:
```python fix_segs.py```

## Step 2 (optional): Creating your own models
If you want to use one of the provided models, skip the remainder of Step 2 and go to Step 3.1.

### Step 2.1: Apply style transforms
Please refer to the report for more information on style transforms. As of now, the style transforms take a long period of time. As such, this data augmentation techniques is done beforehand and the styled images are saved. To apply the style transforms, switch branch to styled-image-gen. Once you checkout the branch, run the following:

```python3 gen_styled_images.py ROOTDIR```

where ```ROOTDIR``` is the location of the data. The styled images will be saved in a new directory ```styled``` under ```ROOTDIR```.

### Step 2.2: Run the model
The remaining two data augmentation techniques can be computed at runtime. To train the model, switch to the main branch and go to ```model```. Before running any scripts, fill out the ```main.json``` file.
Within ```main.json```:
* ```rootdir```: the location of the data (string).
* ```saved_model_loc```: the location of where to save the model (string). THIS MUST END IN '.pt'.
* ```checkpoint_loc```: the location of where to save the checkpoints (string). THIS MUST END IN '.pt'.
* ```num_epochs```: the total number of epochs to run the training data through the model (int).
* ```batch_size```: the size of each batch of data (int).

Once these values are filled in, run ```main.py``` with some version of the following command:

```CUDA_VISIBLE_DEVICES={GPU_IDS} python -m torch.distributed.launch --nproc_per_node={NUM_GPUS} main.py 1> out.txt 2> err.txt &```

```GPU_IDS``` is a list of the process IDs for the GPUs. For example if there are 8 GPUs available, the IDs will be 0,1,2,3,4,5,6,7. If you want to use 4 GPUs, fill ```GPU_IDS``` with ```0,1,2,3```, where you are using the GPUs with IDs 0,1,2, and 3. For running on small GPU Clusters, ```NUM_GPUS``` is an integer of the total number of GPUs you want to use. Any errors will show in err.txt and the general output from the script will show in out.txt. When running this, be careful of your batch size, as you can easily overload memory.

## Step 3: Creating predictions
This step can be completed with or without Step 2. If you did not run Step 2, please use one of the models in ```pre-trained-models``` as your model. There are three pre-trained models.

1) ```default-model.pt``` is the model trained with no data augmentation techniques.
2) ```traditional-model.pt``` is the model trained with only the traditional augmentation techniques.
3) ```all-trans-model.pt``` is the model trained with all data augmentation techniques (traidtional, overlay, and style).

Please refer to the technical report for more information about the different data augmnetation techniques and the different models provided.

### Step 3.1: Run the predictions
To create the predictions from the trained models, first fill in the ```predict.json``` file. Within ```predict.json```:
* ```rootdir```: location of the prediction data (string).
* ```resultsdir```: location of where the results (predictions versus ground truth) will be saved (string).
* ```saved_model_loc```: location of the model to use for creating predictions

Once these values are filled in, run ```predict.py``` with some version of the following command:

```CUDA_VISIBLE_DEVICES={GPU_IDS} python -m torch.distributed.launch --nproc_per_node={NUM_GPUS} predict.py 1> out.txt 2> err.txt &```

There is not much need to run this on more than one GPU since ```predict.py``` will only run through the data once.

### Step 3.2: Run the evaluation
After Step 3.1, the evaluation (provided by the challenge hosts), can be executed by the following:

```python evaluate.py {RESULTSDIR}```,
where ```RESULTSDIR``` is the string location path of the results directory.

## Acknowledgements
All of our results were computed via small GPU computer clusters provided by Oak Ridge National Laboratory. If you have any questions are concerns, please reach out to any of use via the following emails: gkroiz1@umbc.edu, sanderschulhoff@gmail.com, joshua@utk.edu, or ndrake1@vols.utk.edu.

