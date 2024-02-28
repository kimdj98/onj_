import os
import sys

sys.path.append(os.getcwd())

from data.dataset import patient_dicts, ExtractSliced, LoadJsonLabeld
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import cv2
import numpy as np

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


def get_split(split: str):
    with open(f"/mnt/4TB1/onj/dataset/v0/{split}.txt", "r") as f:
        patients = []
        split = f.readlines()
        for line in split:
            patients.append(line.split(" ")[0].split("/")[1])

        return patients


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def create_yolo_dataset(cfg: DictConfig):
    dim_x = cfg.data.CT_dim[0]
    dim_y = cfg.data.CT_dim[1]

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            Flipd(keys=["image"], spatial_axis=2),  # Flip the image along the z-axis
            ExtractSliced(keys=["image"]),
            LoadJsonLabeld(keys=["label"]),  # Use the custom transform for labels
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=3000, b_min=0.0, b_max=1.0, clip=True),
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
            # ),
            Rotate90d(keys=["image"], spatial_axes=(0, 1)),
            Resized(keys=["image"], spatial_size=(dim_x, dim_y, -1), mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    )
    # data_dicts = test_load_data_dict()
    patientdicts = patient_dicts(cfg)
    dataset = Dataset(data=patientdicts, transform=transforms)

    train = get_split("train")
    val = get_split("val")
    test = get_split("test")

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
            img_slice = np.array(img[..., i])
            img_slice = img_slice.transpose(1, 2, 0) * 255
            label_slice = label[i, :]
            # write .jpg file in file_name.jpg
            cv2.imwrite(f"/mnt/4TB1/onj/dataset/v0/YOLO/images/{split}/{file_name}_{i}.jpg", img_slice)
            # write .txt file in yolo format
            with open(f"/mnt/4TB1/onj/dataset/v0/YOLO/labels/{split}/{file_name}_{i}.txt", "w") as f:
                if label_slice[0] == 1.0:
                    f.write(f"{0} {label_slice[1]} {label_slice[2]} {label_slice[3]} {label_slice[4]}\n")
                elif label_slice[0] == 0.0:
                    f.write("")
                else:
                    raise ValueError("label should be 0 or 1")


if __name__ == "__main__":
    create_yolo_dataset()
