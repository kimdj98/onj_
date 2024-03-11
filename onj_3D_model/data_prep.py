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

def get_split(split: str):
    with open(f"/mnt/4TB1/onj/dataset/v0/{split}.txt", "r") as f:
        patients = []
        split = f.readlines()
        for line in split:
            patients.append(line.split(" ")[0].split("/")[1])

        return patients


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg:DictConfig):

    patientdicts = patient_dicts(cfg)
    dataset = Dataset(data=patientdicts, transform=None)


    print(patientdicts)
    quit()

    # train = get_split("train")
    # val = get_split("val")
    # test = get_split("test")




if __name__ == "__main__":
    main()





#### TODO change into creating 3d dataset (slices)
# dataloader = DataLoader(dataset, batch_size=1, num_workers=4)
for patient in tqdm(dataset):
    modal_dir = patient["name"].name
    date = patient["name"].parent.name
    patient_name = patient["name"].parent.parent.parent.name
    file_name = patient_name + "_" + date + "_" + modal_dir

    # find which split the patient belongs to
    if patient_name in train:
        split = "train"
    elif patient_name in val:
        split = "val"
    elif patient_name in test:
        split = "test"
    else:
        print("patient not in split file")
        continue

    num_slices = patient["image"].shape[-1]

    # write splits to YOLO file
    img = patient["image"]
    label = patient["label"]

    for i in range(num_slices):
        img_slice = np.array(img[..., i]) * 255
        # img_slice = img_slice.transpose(1, 2, 0)
        img_slice = img_slice.astype(np.uint8)
        label_slice = label[i, :]
        # write .txt file in yolo format
        with open(f"{BASE_PATH}/{DATA_PATH}/labels/{split}/{file_name}_{i}.txt", "w") as f:
            if label_slice[0] == 1.0:
                # write .jpg file in file_name.jpg
                cv2.imwrite(f"{BASE_PATH}/{DATA_PATH}/images/{split}/{file_name}_{i}.jpg", img_slice[0])
                f.write(f"{0} {label_slice[1]} {label_slice[2]} {label_slice[3]} {label_slice[4]}\n")
            elif label_slice[0] == 0.0:
                pass
            else:
                raise ValueError("label should be 0 or 1")