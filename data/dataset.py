# dataset.py
# TODO: Add the code to load the data from the split and create a dictionary of the data
import hydra
import torch
import json
import os
from pathlib import Path
from omegaconf import DictConfig
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

import torch
# from torch.utils.data import Dataset, DataLoader

class Modal(Enum):
    MDCT = "MDCT"
    CBCT = "CBCT"
    PA = "panorama"


class Direction(Enum):
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"


def load_data_dict(conf: DictConfig, modal: str, dir: str, type: str) -> dict:
    data_dicts = []
    DATA_PATH = Path(conf.data.data_dir)
    for patient in DATA_PATH.glob(f"{type}/*"):
        if (patient / modal).exists():
            for modal_dir in (patient / modal).glob("*/*"):
                if not modal_dir.is_dir():
                    continue
                if "nifti" in os.listdir(modal_dir) and "label.json" in os.listdir(modal_dir) and dir in modal_dir.name:
                    data_dicts.append(
                        {
                            "image": modal_dir / "nifti" / "output.nii.gz",
                            "label": modal_dir / "label.json",
                        }
                    )
    return data_dicts


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def test_load_data_dict(conf: DictConfig):
    data_dict_MDCT_axial = load_data_dict(conf, modal="MDCT", dir="axial", type="ONJ_labeling")
    data_dict_CBCT_axial = load_data_dict(conf, modal="CBCT", dir="axial", type="ONJ_labeling")
    data_dict_MDCT_coronal = load_data_dict(conf, modal="MDCT", dir="coronal", type="ONJ_labeling")
    data_dict_CBCT_coronal = load_data_dict(conf, modal="CBCT", dir="coronal", type="ONJ_labeling")


class ExtractSliced(MapTransform):
    """
    Custom transform to extract slices using SOI information from a 3D image.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        label_path = data["label"]

        if label_path == "":
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


def helper_function(patient, modal, data_dicts, include_name=False):
    """
    helper function for function patient_dicts
    1. check if patient has label.json(check if it has labels)
    2. check if patient has certain modal ("MDCT", "CBCT")
    3. if the patient has certain modal add the nifti path and labels to data_dicts
    """
    if (patient / modal).exists():
        if modal == Modal.CBCT or modal == Modal.MDCT:  # case: CT
            for modal_dir in (patient / modal).glob("*/*"):
                if not modal_dir.is_dir():
                    continue
                if (
                    "nifti" in os.listdir(modal_dir)
                    and "label.json" in os.listdir(modal_dir)
                    and "axial" in modal_dir.name
                ):
                    if include_name:
                        data_dicts.append(
                            {
                                "name": modal_dir,
                                "image": modal_dir / "nifti" / "output.nii.gz",
                                "label": modal_dir / "label.json",
                            }
                        )
                    else:
                        data_dicts.append(
                            {
                                "image": modal_dir / "nifti" / "output.nii.gz",
                                "label": modal_dir / "label.json",
                            }
                        )

        if modal == Modal.PA:
            for pa_img in (patient / modal).glob("*/*"):
                if include_name:
                    data_dicts.append(
                        {
                            "name": modal_dir,
                            "image": pa_img,
                            "label": modal_dir / "label.json",
                        }
                    )
                else:
                    data_dicts.append(
                        {
                            "image": modal_dir / "nifti" / "output.nii.gz",
                            "label": modal_dir / "label.json",
                        }
                    )


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def patient_dicts(cfg: DictConfig) -> Dataset:

    # create data_dicts
    DATAPATH = Path(cfg.data.ONJ_dir)
    include_name = cfg.data.data_generation
    data_dicts = []

    for patient in DATAPATH.glob("*"):
        if (patient / "label.json").exists():
            helper_function(patient, "CBCT", data_dicts, include_name=include_name)
            helper_function(patient, "MDCT", data_dicts, include_name=include_name)

        else:
            print(patient)

    return data_dicts


class PA_dataset(torch.utils.data.Dataset):
    


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    dim_x = cfg.data.CT_dim[0]
    dim_y = cfg.data.CT_dim[1]

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

    # data_dicts = test_load_data_dict()
    patientdicts = patient_dicts(cfg)
    dataset = Dataset(data=patientdicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, num_workers=4)

    import cv2
    import numpy as np

    slice_num = 67
    sample_image = dataset[0]["image"][..., slice_num]
    sample_label = dataset[0]["label"][slice_num, :]
    (x, y, w, h) = np.array(sample_label[1:]) * 512

    # print("Image data:", np.array(sample_image))

    # Correct the data type and shape
    im = np.array(sample_image) * 255
    im = im.astype(np.uint8)  # Ensure data type is np.uint8

    # Debug: Print shape and data type to confirm
    print("Image shape:", im.shape)
    print("Image data type:", im.dtype)

    # Draw the rectangle
    copied_image = im.copy()  # for debugging: don't know why it works...

    cv2.rectangle(
        copied_image[0], (int(x - (w / 2)), int(y - (h / 2))), (int(x + (w / 2)), int(y + (h / 2))), (255,), 2
    )

    # cv2.imshow("image", copied_image[0])

    # Save the image
    cv2.imwrite("sa.png", copied_image[0])


if __name__ == "__main__":
    main()
