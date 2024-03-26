import os
import sys

sys.path.append(os.getcwd())

from data.dataset import patient_dicts, ExtractSliced, LoadJsonLabeld
import hydra
from omegaconf import DictConfig
from tqdm import tqdm
import cv2
import numpy as np
from enum import Enum
from pathlib import Path
import json

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


# Enum class
class Modal(Enum):
    CT = "CT"
    PA = "panorama"


class Direction(Enum):
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def create_yolo_dataset(cfg: DictConfig):
    modal = cfg.data.modal
    # check modal
    if modal == "CT":
        modal = Modal.CT
    elif modal == "panorama":
        modal = Modal.PA
    else:
        raise ValueError("modal should be CT or PA")

    if modal == Modal.CT:
        BASE_PATH = cfg.data.data_dir
        DATA_PATH = f"YOLO_with_labels"
        if not os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/test")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/test")

        dim_x = cfg.data.CT_dim[0]
        dim_y = cfg.data.CT_dim[1]

        transforms = Compose(
            [
                LoadImaged(keys=["image"]),
                EnsureChannelFirstd(keys=["image"]),
                Flipd(keys=["image"], spatial_axis=2),  # Flip the image along the z-axis
                ExtractSliced(keys=["image"]),
                LoadJsonLabeld(keys=["label"]),  # Use the custom transform for labels
                # ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=3000, b_min=0, b_max=1, clip=True),
                ScaleIntensityRangePercentilesd(
                    keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=True, relative=False
                ),
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

    elif modal == Modal.PA:
        BASE_PATH = cfg.data.data_dir
        DATA_PATH = f"YOLO_PA"
        if os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
            print("YOLO_PA already exists")

        if not os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/test")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/test")

        # dim_x = cfg.data.PA_dim[0]
        # dim_y = cfg.data.PA_dim[1]

        def inner(split: str = "train"):
            base_dir = Path(cfg.data.data_dir)
            # read train.txt
            with open(f"{base_dir}/{split}.txt", "r") as f:
                patients = f.readlines()
                patients = [line.split(" ")[0].split("/")[1] for line in patients]

            for patient in tqdm(patients):
                patient_dir = base_dir / "ONJ_labeling" / patient
                # if patient_dir has no panorama folder, skip
                if not (patient_dir / "panorama").exists():
                    continue

                # if patient_dir has no label.json, skip
                if not (patient_dir / "panorama" / "label.json").exists():
                    continue

                # if patient_dir has no images, skip
                if len(os.listdir(patient_dir / "panorama")) == 0:
                    continue

                # move image to YOLO_PA_train/images
                os.system(f"cp {patient_dir}/panorama/*.jpg {base_dir}/YOLO_PA/images/{split}/{patient_dir.name}.jpg")

                # move label.json to YOLO_PA_train/labels
                # os.system(
                #     f"cp {patient_dir}/panorama/label.json {base_dir}/YOLO_PA/labels/{split}/{patient_dir.name}.json"
                # )

                # write label.txt in YOLO format
                with open(f"{base_dir}/YOLO_PA/labels/{split}/{patient_dir.name}.txt", "w") as f:
                    with open(patient_dir / "panorama" / "label.json", "r") as file:
                        labels = json.load(file)
                        for box in labels["bbox"]:
                            x = box["coordinates"][0]
                            y = box["coordinates"][1]
                            w = box["coordinates"][2]
                            h = box["coordinates"][3]

                            f.write(f"{0} {x} {y} {w} {h}\n")

        inner("train")
        print("Create YOLO_PA_train done")

        inner("val")
        print("Create YOLO_PA_val done")

        inner("test")
        print("Create YOLO_PA_test done")


if __name__ == "__main__":
    create_yolo_dataset()
