import os
import sys

sys.path.append(os.getcwd())

from data_func import load_data_dict, patient_dicts
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import cv2
import numpy as np
from enum import Enum

from monai.data import Dataset, DataLoader
from monai.transforms import (
    Compose,
    MapTransform,
    Transform,
    LoadImaged,
    ScaleIntensityRanged,
    ScaleIntensityRangePercentilesd,
    ToTensord,
    EnsureChannelFirstd,
    Rotate90d,
    Flipd,
    Resized,
)
from monai.config.type_definitions import NdarrayTensor
from monai.visualize.utils import blend_images, matshow3d  ## label과 Image를 합친 영상  ## 3d image의 visulization

import nibabel as nib
from scipy.ndimage import zoom

# data_dict_non_ONJ_MDCT_axial = load_data_dict(conf, modal="MDCT", dir="axial", type="Non_ONJ")
# data_dict_non_ONJ_MDCT_coronal = load_data_dict(conf, modal="MDCT", dir="coronal", type="Non_ONJ")
# data_dict_non_ONJ_MDCT_sagittal = load_data_dict(conf, modal="MDCT", dir="sagittal", type="Non_ONJ")## for 3D U-Net input


# data_dict_non_ONJ_CBCT_axial = load_data_dict(conf, modal="CBCT", dir="axial", type="Non_ONJ")
# data_dict_non_ONJ_CBCT_coronal = load_data_dict(conf, modal="CBCT", dir="coronal", type="Non_ONJ")
# data_dict_non_ONJ_CBCT_sagittal = load_data_dict(conf, modal="CBCT", dir="sagittal", type="Non_ONJ")


# data_dict_ONJ_MDCT_axial = load_data_dict(conf, modal="MDCT", dir="axial", type="ONJ_labeling")
# data_dict_ONJ_MDCT_coronal = load_data_dict(conf, modal="MDCT", dir="coronal", type="ONJ_labeling")
# data_dict_ONJ_MDCT_sagittal = load_data_dict(conf, modal="MDCT", dir="sagittal", type="ONJ_labeling")

# data_dict_ONJ_CBCT_axial = load_data_dict(conf, modal="CBCT", dir="axial", type="ONJ_labeling")
# data_dict_ONJ_CBCT_coronal = load_data_dict(conf, modal="CBCT", dir="coronal", type="ONJ_labeling")
# data_dict_ONJ_CBCT_sagittal = load_data_dict(conf, modal="CBCT", dir="sagittal", type="ONJ_labeling")


save_dir = 'dataset_processed/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_split(split: str):
    with open(f"/mnt/4TB1/onj/dataset/v0/{split}.txt", "r") as f:
        patients = []
        split = f.readlines()
        for line in split:
            patients.append(line.split(" ")[0].split("/")[1])

        return patients


transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Flipd(keys=["image"], spatial_axis=2),  # Flip the image along the z-axis
        ExtractSliced(keys=["image"]),
        LoadJsonLabeld(keys=["label"]),  # Use the custom transform for labels
        # ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        ScaleIntensityRangePercentilesd(
            keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
        ),  # emperically known to be better than ScaleIntensityRanged
        Rotate90d(keys=["image"], spatial_axes=(0, 1)),
        Resized(keys=["image"], spatial_size=(dim_x, dim_y, -1), mode="trilinear"),
        ToTensord(keys=["image"]),
    ]
)

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg:DictConfig):

    patientdicts = patient_dicts(cfg)
    dataset = Dataset(data=patientdicts, transform=None)

    train = get_split("train")
    val = get_split("val")
    test = get_split("test")

    train_total = []
    val_total = []
    test_total = []

    train_label = []
    val_label = []
    test_label = []

    count_train = 0

    for patient in tqdm(dataset):

        modal_dir = patient["name"].name
        date = patient["name"].parent.name
        patient_name = patient["name"].parent.parent.parent.name
        file_name = patient_name + "_" + date + "_" + modal_dir

        ONJ_class = patient["name"].parent.parent.parent.parent.name

        # find which split the patient belongs to
        if patient_name in train:
            split = "train"
            data_total = train_total 
            label_total = train_label
            count_train += 1
        elif patient_name in val:
            split = "val"
            data_total = val_total 
            label_total = val_label 
        elif patient_name in test:
            split = "test"
            data_total = test_total 
            label_total = test_label
        else:
            print("patient not in split file")
            continue
        
        img_obj = nib.load(patient["image"]) ## (512, 512, 190)
        img_3d = img_obj.get_fdata()

        #convert to channel-first numpy array # for 3D U-Net input
        img_3d = np.moveaxis(img_3d, -1, 0)
        
        ## get label data  ## classification label
        if "ONJ_labeling" in ONJ_class:
            label = 1
        elif "Non_ONJ" in ONJ_class:
            label = 0
        else:
            print(ONJ_class)


        ## image preprocessing
        depth_ratio = 120 / img_3d.shape[0] #desired depth = 120
        wh_ratio = 512 / img_3d.shape[1] #desired depth = 256

        resliced_img_3d = zoom(img_3d, (depth_ratio, 1, 1))
        resized_img_3d = zoom(resliced_img_3d, (1, wh_ratio, wh_ratio))

        print(np.max(resliced_img_3d), np.min(resliced_img_3d))
        quit()

        data_total.append(resized_img_3d)
        label_total.append(label)

        test = resized_img_3d[50, :, :]



    train_total = np.stack(train_total, axis=0)
    train_label = np.stack(train_label, axis=0)

    val_total = np.stack(val_total, axis=0)
    val_label = np.stack(val_label, axis=0)
    
    test_total = np.stack(test_total, axis=0)
    test_label = np.stack(test_label, axis=0)

    print(train_total.shape)
    print(val_total.shape)
    print(test_total.shape)

    np.save(save_dir+'train_total')
    quit()



if __name__ == "__main__":
    main()




