# ONJ Classification using 3D U-Net

ONJ classification Using 3D U-Net structure

## Dataset Preprocessing
- Run the script to make preprocessed datasets.
- We use the original dataset placed in mnt folder in DGX server
- We do not reshape or reslice images here, but in training process for producing experiments with different hyperparameters regarding resizing or reslicing
  
```sh
python data_prep.py 
```
## Model Training 
- Run the script to train the model.
- GPU number can be given as argument.
- For training, use mode 1, and for testing, use mode 2
- Define experiment number which will be used for saving model
- Results of testing which shows ROC curves and sample iamges are saved in '{EXP_name}_test' folder
```sh
python train.py -g [gpu number] -m [mode] -exp [experiment name]
```

