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
import json
import torch

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

size = 512 #size for image reshaping 
save_dir = f'dataset_processed_{str(size)}/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)


def get_split(split: str):
    with open(f"/mnt/4TB1/onj/dataset/v0/{split}.txt", "r") as f:
        patients = []
        label = []
        split = f.readlines()
        for line in split:
            patients.append(line.split(" ")[0].split("/")[1])
            label.append(int(line.split(" ")[1]))

        return patients, label
class ExtractSliced(MapTransform):
    """
    Custom transform to extract slices using SOI information from a 3D image.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        label_path = data["label"]

        if label_path == "" or "Non_ONJ" in data["name"].parent.parent.parent.parent.name or \
        os.path.exists(label_path)==False:
            return data


        with open(label_path, "r") as file:
            labels = json.load(file)
            try:
                SOI = labels["SOI"]
                data["image"] = data["image"][..., SOI[0] : SOI[1]]
            # except:
            #     # use all slices if SOI is not available
            #     pass
            finally:
                return data


class LoadJsonLabeld(MapTransform):
    """
    Custom transform to load bounding box coordinates from a JSON file.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False, x_dim: int = 512, y_dim: int = 512):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, data: dict) -> dict:
        total_slices = data["image"].shape[-1]

        label_path = data["label"]
        if label_path == "":
            t = torch.zeros((total_slices, 5))
            data["label"] = t
            return data
        onj_class = label_path.parent.parent.parent.parent.parent.name
        if "Non_ONJ" in onj_class:
            t = torch.zeros((total_slices, 5))
            data["label"] = t
            return data



        with open(label_path, "r") as file:
            labels = json.load(file)

        num_labels = len(labels["slices"])

        try:
            SOI = labels["SOI"]
        except:
            SOI = [0, 0]
        label_start = labels["slices"][0]["slice_number"] - SOI[0]
        label_end = labels["slices"][-1]["slice_number"] - SOI[0]

        t = torch.zeros((total_slices, 5), dtype=torch.float32)

        for i in range(num_labels):
            slice = labels["slices"][i]
            slice_number = slice["slice_number"] - SOI[0]
            # TODO: Check if the coordinate type is Cx, Cy, w, h
            x = slice["bbox"][0]["coordinates"][0]
            y = slice["bbox"][0]["coordinates"][1]
            w = slice["bbox"][0]["coordinates"][2]
            h = slice["bbox"][0]["coordinates"][3]

            t[slice_number] = torch.tensor([1.0, x, y, w, h])

        data["label"] = t

        return data

transforms = Compose(
    [
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Flipd(keys=["image"], spatial_axis=2),  # Flip the image along the z-axis
        
        
        ## resolved Non_ONJ SOI problem
        ExtractSliced(keys=["image"]),
        
        
        LoadJsonLabeld(keys=["label"]),  # Use the custom transform for labels


        # ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
        # ScaleIntensityRangePercentilesd(
        #     keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
        # ),  # emperically known to be better than ScaleIntensityRanged
        # Rotate90d(keys=["image"], spatial_axes=(0, 1)),
        # Resized(keys=["image"], spatial_size=(dim_x, dim_y, -1), mode="trilinear"),
        # ToTensord(keys=["image"]),
    ]
)

@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg:DictConfig):

    patientdicts = patient_dicts(cfg)
    dataset = Dataset(data=patientdicts, transform=transforms)


    train, _ = get_split("train")
    val, _ = get_split("val")
    test, _ = get_split("test")

    train_label = []
    val_label = []
    test_label = []

    train_total_org = {}
    val_total_org = {}
    test_total_org = {}

    train_total_seg_npy = {}
    val_total_seg_npy = {}
    test_total_seg_npy = {}

    count_train = 0
    count_val = 0
    count_test = 0
    count_onj = 0
    count_nonj = 0

    for patient in tqdm(dataset):


        modal_dir = patient["name"].name
        date = patient["name"].parent.name
        patient_name = patient["name"].parent.parent.parent.name
        file_name = patient_name + "_" + date + "_" + modal_dir

        ONJ_class = patient["name"].parent.parent.parent.parent.name

        print(patient_name)
        if str(ONJ_class)=='Non_ONJ' or str(ONJ_class)=="ONJ_labeling":

            # find which split the patient belongs to
            if patient_name in train:
                split = "train"

                label_total = train_label

                count_train += 1
                count = count_train
                data_total_org = train_total_org
                data_total_seg_npy = train_total_seg_npy

            elif patient_name in val:
                split = "val"

                label_total = val_label

                count_val += 1
                count = count_val
                data_total_org = val_total_org
                data_total_seg_npy = val_total_seg_npy

            elif patient_name in test:
                split = "test"

                label_total = test_label

                count_test += 1
                count = count_test
                data_total_org = test_total_org
                data_total_seg_npy = test_total_seg_npy

            else:
                print("patient not in split file")
                continue

            ## uncomment below line when not using transforms
            # img_obj = nib.load(patient["image"]) ## (512, 512, 190)
            # img_3d = img_obj.get_fdata()

            img_3d = patient["image"].squeeze(0) # (512, 512, 91)

            #convert to channel-first numpy array # for 3D U-Net input
            img_3d = np.moveaxis(img_3d, -1, 0) #(91, 512, 512)

            if str(ONJ_class) == 'ONJ_labeling':

                ## label segmentation
                seg_3d_label = patient["label"]

                ## Also save bounding box information as array
                seg_npy = np.zeros((img_3d.shape[0], 4))

                for slice_num in range(seg_3d_label.shape[0]):
                    onj_exist = seg_3d_label[slice_num][0]

                    x,y,w,h = seg_3d_label[slice_num][1:] * img_3d.shape[1]

                    ## consider original image size
                    x = int(seg_3d_label[slice_num][1]*img_3d.shape[2])
                    y = int(seg_3d_label[slice_num][2]*img_3d.shape[1])
                    w = int(seg_3d_label[slice_num][3]*img_3d.shape[2])
                    h = int(seg_3d_label[slice_num][4]*img_3d.shape[1])

                    x,y,w,h = seg_3d_label[slice_num][1:]
                    
                    seg_npy[slice_num] = [x,y,w,h]

                    if int(onj_exist) == 1:
                        seg_3d_mask[slice_num, y:y+h, x:x+w] = 1
                        
            elif str(ONJ_class) == 'Non_ONJ':
                seg_3d_mask = np.zeros((img_3d.shape), dtype=np.uint8)
                seg_npy = np.zeros((img_3d.shape[0], 4), dtype=np.uint8) + 9999
                

            
            ## get label data  ## classification label
            if "ONJ_labeling" in ONJ_class:
                label = 1
                count_onj += 1
            elif "Non_ONJ" in ONJ_class:
                label = 0
                count_nonj += 1
            else:
                print(ONJ_class)

            print('original size: ', img_3d.shape, ONJ_class)

            data_total_seg_npy[f'{count}'] = seg_npy
            data_total_org[f'{count}'] = img_3d

            ##! DO NOT PREPROCESS IMAGE HERE
            ##! PREPROCESSING IS AT train.py, and we have to save the original arrays without resizing 
            ##! for experimental trials regarding hyperparameters for resizing and reslicing
            # depth_ratio = 64 / img_3d.shape[0] #desired depth = 70 (empirically chosen)
            # wh_ratio1 = size / img_3d.shape[1] #desired depth = size (empirically chosen)
            # wh_ratio2 = size / img_3d.shape[2]
            # resliced_img_3d = zoom(img_3d, (depth_ratio, 1, 1))
            # resized_img_3d = zoom(resliced_img_3d, (1, wh_ratio1, wh_ratio2))
            # resliced_seg_3d = zoom(seg_3d_mask, (depth_ratio, 1, 1))
            # resized_seg_3d = zoom(resliced_seg_3d, (1, wh_ratio1, wh_ratio2))
            #!##############################
            label_total.append(label)

        print(count_train, count_val, count_test)


    train_label = np.stack(train_label, axis=0)

    val_label = np.stack(val_label, axis=0)
    
    test_label = np.stack(test_label, axis=0)

    np.save(save_dir+'train_label.npy', train_label)
    np.save(save_dir+'val_label.npy', val_label)
    np.save(save_dir+'test_label.npy', test_label)

    # # Save all scans in a single .npz file
    np.savez(save_dir+"/train_total_org.npz", **train_total_org)
    np.savez(save_dir+"/val_total_org.npz", **val_total_org)
    np.savez(save_dir+"/test_total_org.npz", **test_total_org)
    
    np.savez(save_dir+'/train_total_seg_org.npz', **train_total_seg_org)
    np.savez(save_dir+'/val_total_seg_org.npz', **val_total_seg_org)
    np.savez(save_dir+'/test_total_seg_org.npz', **test_total_seg_org)

    # quit()



if __name__ == "__main__":
    main()




