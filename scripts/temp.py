import monai
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import os
import json

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

# from ultralytics import YOLO
from tqdm import tqdm

# model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n detection model


def add_data_dicts(data_dicts: list, DATAPATH: Path, ONJ: bool = True):
    def helper(Patient:Path, data_dicts: list[Path], ONJ: bool, modal: str):
        has_modal = False
        if (Patient / modal).exists():
            has_modal = True
            for modal_dir in (Patient / modal).glob("*/*"):
                if not modal_dir.is_dir():
                    continue
                if (
                    "nifti" in os.listdir(modal_dir)
                    and "label.json" in os.listdir(modal_dir)
                    and "axial" in modal_dir.name
                ):
                    labels = json.load(open(modal_dir / "label.json", "r"))

                    if "SOI" in labels.keys():
                        SOI= labels["SOI"]
                    else:
                        SOI = [0,0]

                    data_dicts.append(
                        {
                            "image": modal_dir / "nifti" / "output.nii.gz",
                            "label": modal_dir / "label.json",
                            "SOI": SOI,
                            "ONJ": ONJ,
                        }
                    )

        return has_modal

    num_CBCT = 0
    num_MDCT = 0
    no_modal = 0
    
    for patient in DATAPATH.glob("*"):
        num_CBCT += helper(patient, data_dicts, ONJ, modal="CBCT")
        num_MDCT += helper(patient, data_dicts, ONJ, modal="MDCT")

        if not (patient / "MDCT").exists() and not (patient / "CBCT").exists():
            print(patient)
            no_modal += 1

    # print out results
    print(f"Total patients: {num_CBCT + num_MDCT}, No CT: {no_modal}")

class LoadJsonLabeld(MapTransform):
    """
    Custom transform to load bounding box coordinates from a JSON file if the data has label.json.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False, x_dim: int = 512, y_dim: int = 512):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, data: dict) -> dict:
        total_slices = data["image"].shape[-1]

        t = torch.zeros((total_slices, 5))
        data_path = data["label"]

        if data_path == "":
            data["label"] = t
            return data

        with open(data_path, "r") as file:
            labels = json.load(file)

        # when labels have key slices
        if "slices" in labels.keys():
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

        if data["SOI"] != [0, 0]:
            SOI = data["SOI"]
            t = t[SOI[0]:SOI[1]]

        data["label"] = t

        return data
    
class SelectSliced(MapTransform):
    """
    Custom transform to select slices we are interested from a 3D image
    """
    def __init__(self, keys: str, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        if data["SOI"] == [0, 0]: # SOI not found
            return data
        
        else:
            SOI = data["SOI"]
            data["image"] = data["image"][..., SOI[0]:SOI[1]]
            return data


def viz_intensity_grad(num_slice: int, dataset: Dataset):
    for i in tqdm(range(num_slice)):
        sample_img = dataset[i]["image"][0]
        # read json
        json_path = dataset[i]["label"]
        with open(json_path, "r") as file:
            labels = json.load(file)

        MODAL = labels["folder_path"].split("/")[2]
        patient = labels["folder_path"].split("/")[1]
        try:
            SOI = labels["SOI"]
            num_SOI = SOI[1] - SOI[0]
        except:
            SOI = "SOI not found"
            num_SOI = 0

        slice_brightness = sample_img.sum((0, 1)).unsqueeze(0).unsqueeze(0)               # shape: (1, 1, slice)
        # Convolute 1D gradient kernel using torch
        kernel = torch.tensor([-1, 0, 1], dtype=torch.float32).unsqueeze(0).unsqueeze(0)  # shape: (1, 1, 3)
        
        slice_gradient = F.conv1d(slice_brightness, kernel, padding=1)                   # shape: (1, 1, slice)
        # draw slice_brightness graph and slice_gradient in one graph
        fig, ax = plt.subplots(2, 1, figsize=(10, 10))
        ax[0].plot(slice_brightness[0, 0].numpy())
        ax[1].plot(slice_gradient[0, 0][1:-1].numpy())
        ax[0].set_title(f"Slice Brightness of {MODAL}, SOI: {SOI}, num_SOI: {num_SOI}")
        ax[1].set_title(f"Slice Gradient of {MODAL}, SOI: {SOI}, num_SOI: {num_SOI}")

        # draw vertical line in the graph if SOI is a list
        if SOI != "SOI not found":
            ax[0].axvline(x=SOI[0], color='r', linestyle='-')
            ax[0].axvline(x=SOI[1], color='r', linestyle='-')
            ax[1].axvline(x=SOI[0], color='r', linestyle='-')
            ax[1].axvline(x=SOI[1], color='r', linestyle='-')

        # save the fig
        plt.savefig(f"{i}_{patient}.png")
        plt.close(fig)


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    dim_x = cfg.data.CT_dim[0]
    dim_y = cfg.data.CT_dim[1]

    transforms = Compose(
        [
            LoadImaged(keys=["image"]),
            EnsureChannelFirstd(keys=["image"]),
            LoadJsonLabeld(keys=["label"]),  # Use the custom transform for labels
            ScaleIntensityRanged(keys=["image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
            # ),
            Rotate90d(keys=["image"], spatial_axes=(0, 1)),
            Flipd(keys=["image"], spatial_axis=2),
            SelectSliced(keys=["image", "SOI"]),
            Resized(keys=["image"], spatial_size=(dim_x, dim_y, 64), mode="trilinear"),
            ToTensord(keys=["image"]),
        ]
    )

    # create data_dicts
    ONJPATH = Path(cfg.data.ONJ_dir)
    NON_ONJPATH = Path(cfg.data.NON_ONJ_dir)
    data_dicts = []

    add_data_dicts(data_dicts, ONJPATH, ONJ=True, includes=["CBCT", "MDCT"])
    add_data_dicts(data_dicts, NON_ONJPATH, ONJ=False, includes=["CBCT", "MDCT"])
    # add_data_dicts(data_dicts, ONJPATH, ONJ=True)
    # add_data_dicts(data_dicts, NON_ONJPATH, ONJ=False)

    # Create a MONAI dataset
    dataset = Dataset(data=data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    print(f"Total patients: {dataset.__len__()}")

    for (i, data) in enumerate(dataloader):
        print(data["image"].shape)
        print(data["label"].shape)
        print(data["SOI"])
        print(data["ONJ"])
        print("=====================================")
        if i == 10:
            break
    # viz_intensity_grad(100, dataset)

    # import cv2
    # # write slices
    # for i in tqdm(range(64)):
    #     cv2.imwrite(f"slice_{i+1}.png", (np.array(dataset[0]["image"][:,:,:,i])*255).astype(np.uint8)[0])


    # slice_num = 150
    # sample_image = dataset[0]["image"][..., slice_num]
    # sample_label = dataset[0]["label"][slice_num, :]
    # (x, y, w, h) = np.array(sample_label[1:]) * 512

    # # print("Image data:", np.array(sample_image))

    # # Correct the data type and shape
    # im = np.array(sample_image) * 255
    # im = im.astype(np.uint8)  # Ensure data type is np.uint8

    # # Debug: Print shape and data type to confirm
    # print("Image shape:", im.shape)
    # print("Image data type:", im.dtype)

    # # Draw the rectangle
    # copied_image = im.copy()  # for debugging: don't know why it works...

    # cv2.rectangle(
    #     copied_image[0], (int(x - (w / 2)), int(y - (h / 2))), (int(x + (w / 2)), int(y + (h / 2))), (255,), 2
    # )

    # # cv2.imshow("image", copied_image[0])

    # # Save the image
    # cv2.imwrite("sa.png", copied_image[0])

    # pass

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
