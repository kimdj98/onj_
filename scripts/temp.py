import monai
import torch
import numpy as np
import os

from pathlib import Path

import hydra
from omegaconf import DictConfig

# from preprocess.bbox_normalizer import normalize
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

from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model


class LoadJsonLabel(MapTransform):
    """
    Custom transform to load bounding box coordinates from a JSON file.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False, x_dim: int = 512, y_dim: int = 512):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, data: dict) -> dict:
        total_slices = data["image"].shape[-1]

        data_path = data["label"]
        if data_path == "":
            t = torch.zeros((total_slices, 5))
            data["label"] = t
            return data

        with open(data_path, "r") as file:
            labels = json.load(file)

        num_labels = len(labels["slices"])
        label_start = labels["slices"][0]["slice_number"]
        label_end = labels["slices"][-1]["slice_number"]

        t = torch.zeros((total_slices, 5), dtype=torch.float32)

        for i in range(num_labels):
            slice = labels["slices"][i]
            slice_number = slice["slice_number"]
            # TODO: Check if the coordinate type is x, y, w, h
            x = slice["bbox"][0]["coordinates"][0]
            y = slice["bbox"][0]["coordinates"][1]
            w = slice["bbox"][0]["coordinates"][2]
            h = slice["bbox"][0]["coordinates"][3]

            t[slice_number] = torch.tensor([1.0, x, y, w, h])

        data["label"] = t

        return data


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dim_x = cfg.data.CT_dim[0]
    dim_y = cfg.data.CT_dim[1]

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            LoadJsonLabel(keys=["label"]),  # Use the custom transform for labels
            # ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            ScaleIntensityRangePercentilesd(
                keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
            ),
            Rotate90d(keys=["image"], spatial_axes=(0, 1)),
            Flipd(keys=["image"], spatial_axis=2),
            Resized(keys=["image"], spatial_size=(dim_x, dim_y, -1), mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    )

    # create data_dicts
    DATAPATH = Path(cfg.data.ONJ_dir)

    data_dicts = []

    i = 0
    j = 0

    for patient in DATAPATH.glob("*"):
        if (patient / "label.json").exists():
            # print(patient / "label.json")
            i += 1
            if (patient / "CBCT").exists():
                for modal_dir in (patient / "CBCT").glob("*/*"):
                    if not modal_dir.is_dir():
                        continue
                    if (
                        "nifti" in os.listdir(modal_dir)
                        and "label.json" in os.listdir(modal_dir)
                        and "axial" in modal_dir.name
                    ):
                        data_dicts.append(
                            {
                                "image": modal_dir / "nifti" / "output.nii.gz",
                                "label": modal_dir / "label.json",
                            }
                        )

            if (patient / "MDCT").exists():
                for modal_dir in (patient / "MDCT").glob("*/*"):
                    if not modal_dir.is_dir():
                        continue
                    if (
                        "nifti" in os.listdir(modal_dir)
                        and "label.json" in os.listdir(modal_dir)
                        and "axial" in modal_dir.name
                    ):
                        data_dicts.append(
                            {
                                "image": modal_dir / "nifti" / "output.nii.gz",
                                "label": modal_dir / "label.json",
                            }
                        )
        else:
            print(patient)

        j += 1

    # Create a MONAI dataset
    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    import cv2

    slice_num = 150
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

    pass

    # plt = matshow3d(
    #     volume=dataset[0]["image"][..., 1::20],
    #     fig=None,
    #     title="input image",
    #     frame_dim=-1,
    #     show=True,
    #     cmap="gray",
    # )


if __name__ == "__main__":
    main()
