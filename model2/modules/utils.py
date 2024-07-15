import hydra
from pathlib import Path
import json
import torch

from typing import Any, Dict
from monai.data import Dataset, DataLoader
from data.data_dicts import LoadJsonLabeld, SelectSliced, get_data_dicts
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
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Transposed,
)

# configuration for the data
CT_dim_x = 512
CT_dim_y = 512
PA_dim_x = 2048
PA_dim_y = 1024

transforms = Compose(
    [
        LoadImaged(keys=["CT_image"]),
        EnsureChannelFirstd(keys=["CT_image"]),
        ScaleIntensityRanged(
            keys=["CT_image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=False
        ),  # NOTE: check the range
        Rotate90d(keys=["CT_image"], spatial_axes=(0, 1)),
        Flipd(keys=["CT_image"], spatial_axis=2),
        SelectSliced(keys=["CT_image", "CT_SOI"]),
        Resized(keys=["CT_image"], spatial_size=(CT_dim_x, CT_dim_y, 64), mode="trilinear"),
        ToTensord(keys=["CT_image"]),
    ]
)


def preprocess_data(base_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    base_path = Path(base_path)

    device = data["img"].device

    patient = data["im_file"].split("/")[-1].split(".")[0]  # "EW-0001"

    # if base_path/ONJ_labeling/patient/CBCT exists, then it is a CBCT image and the label is 1 for ONJ
    # if base_path/ONJ_labeling/patient/MDCT exists, then it is a MDCT image and the label is 1 for ONJ
    # if base_path/Non_ONJ_soi/patient/CBCT exists, then it is a CBCT image and the label is 0 for non-ONJ
    # if base_path/Non_ONJ_soi/patient/MDCT exists, then it is a MDCT image and the label is 0 for non-ONJ
    if (base_path / "ONJ_labeling" / patient).exists():
        if (base_path / "ONJ_labeling" / patient / "CBCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            data["cls"] = 1
            ct_dir = base_path / "ONJ_labeling" / patient / "CBCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "CBCT_axial" / "nifti" / "output.nii.gz"

        elif (base_path / "ONJ_labeling" / patient / "MDCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            data["cls"] = 1
            ct_dir = base_path / "ONJ_labeling" / patient / "MDCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "MDCT_axial" / "nifti" / "output.nii.gz"

    elif (base_path / "Non_ONJ_soi" / patient).exists():
        if (base_path / "Non_ONJ_soi" / patient / "CBCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            data["cls"] = 0
            ct_dir = base_path / "Non_ONJ_soi" / patient / "CBCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "CBCT_axial" / "nifti" / "output.nii.gz"

        elif (base_path / "Non_ONJ_soi" / patient / "MDCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            data["cls"] = 0
            ct_dir = base_path / "Non_ONJ_soi" / patient / "MDCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "MDCT_axial" / "nifti" / "output.nii.gz"

    annotations = patient_path / "label.json"
    annotations = json.load(open(annotations, "r"))

    if "SOI" in annotations.keys():
        data["CT_SOI"] = annotations["SOI"]
    else:
        data["CT_SOI"] = [0, 0]

    data = transforms(data)
    # change data format to the same device tensors.

    # Move 'CT_image' tensor to the same device
    data["CT_image"] = data["CT_image"].to(device, non_blocking=True).float()

    # Move 'CT_SOI' to the same device and convert to tensor if needed
    if isinstance(data["CT_SOI"], list):
        data["CT_SOI"] = torch.tensor(data["CT_SOI"], device=device, dtype=torch.float32)
    else:
        data["CT_SOI"] = data["CT_SOI"].to(device, non_blocking=True).float()

    # Move 'bboxes' tensor to the same device
    data["bboxes"] = data["bboxes"].to(device, non_blocking=True).float()

    # Move 'batch_idx' tensor to the same device
    data["batch_idx"] = data["batch_idx"].to(device, non_blocking=True).float()

    return data
