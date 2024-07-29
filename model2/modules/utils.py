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
CT_DIM_X = 512
CT_DIM_Y = 512
PA_DIM_X = 2048
PA_DIM_Y = 1024

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
        Resized(keys=["CT_image"], spatial_size=(CT_DIM_X, CT_DIM_Y, 64), mode="trilinear"),
        ToTensord(keys=["CT_image"]),
    ]
)


def preprocess_data(base_path: Path, data: Dict[str, Any]) -> Dict[str, Any]:
    base_path = Path(base_path)
    for key in data:
        if isinstance(data[key], list):
            data[key] = data[key][0]

    device = data["img"].device
    data["img"] = data["img"].bfloat16()
    patient = data["im_file"].split("/")[-1].split(".")[0]  # "EW-0001"

    # if base_path/ONJ_labeling/patient/CBCT exists, then it is a CBCT image and the label is 1 for ONJ
    # if base_path/ONJ_labeling/patient/MDCT exists, then it is a MDCT image and the label is 1 for ONJ
    # if base_path/Non_ONJ_soi/patient/CBCT exists, then it is a CBCT image and the label is 0 for non-ONJ
    # if base_path/Non_ONJ_soi/patient/MDCT exists, then it is a MDCT image and the label is 0 for non-ONJ
    if (base_path / "ONJ_labeling" / patient).exists():
        data["onj_cls"] = 1
        if (base_path / "ONJ_labeling" / patient / "CBCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            ct_dir = base_path / "ONJ_labeling" / patient / "CBCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "CBCT_axial" / "nifti" / "output.nii.gz"
            data["CT_modal"] = "CBCT"

        elif (base_path / "ONJ_labeling" / patient / "MDCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            ct_dir = base_path / "ONJ_labeling" / patient / "MDCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "MDCT_axial" / "nifti" / "output.nii.gz"
            data["CT_modal"] = "MDCT"

        else:  # if patient has no CT image (exception handler)
            return None

    elif (base_path / "Non_ONJ_soi" / patient).exists():
        data["onj_cls"] = 0
        if (base_path / "Non_ONJ_soi" / patient / "CBCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            ct_dir = base_path / "Non_ONJ_soi" / patient / "CBCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "CBCT_axial" / "nifti" / "output.nii.gz"
            data["CT_modal"] = "CBCT"

        elif (base_path / "Non_ONJ_soi" / patient / "MDCT").exists():
            patient_path = base_path / "ONJ_labeling" / patient
            ct_dir = base_path / "Non_ONJ_soi" / patient / "MDCT"
            ct_date_dir = list(ct_dir.glob("*"))[0]
            data["CT_image"] = ct_date_dir / "MDCT_axial" / "nifti" / "output.nii.gz"
            data["CT_modal"] = "MDCT"

        else:  # if patient has no CT image (exception handler)
            return None

    else:  # if patient not in both labels
        raise ValueError(f"Patient {patient} is not in both labels")

    try:  # if the patient has no label.json, then return None
        annotations = patient_path / "label.json"
        annotations = json.load(open(annotations, "r"))

    except:
        annotations = ct_date_dir / f"{data['CT_modal']}_axial" / "label.json"
        annotations = json.load(open(annotations, "r"))

    if "SOI" in annotations.keys():
        data["CT_SOI"] = annotations["SOI"]
    else:
        data["CT_SOI"] = [0, 0]

    data = transforms(data)
    # change data format to the same device tensors.

    if data["CT_modal"] == "MDCT":
        # reverse the order of the slices because the MDCT images are in reverse order
        data["CT_image"] = torch.flip(data["CT_image"], dims=[-1])

    # Move 'CT_image' tensor to the same device
    data["CT_image"] = (
        data["CT_image"].to(device, non_blocking=True).bfloat16().unsqueeze(0)
    )  # NOTE: for originally the channel size is 1 need additional batch expansion

    # Move 'CT_SOI' to the same device and convert to tensor if needed
    if isinstance(data["CT_SOI"], list):
        data["CT_SOI"] = torch.tensor(data["CT_SOI"], device=device, dtype=torch.float32)
    else:
        data["CT_SOI"] = data["CT_SOI"].to(device, non_blocking=True).float()

    # Move 'bboxes' tensor to the same device
    data["bboxes"] = data["bboxes"].to(device, non_blocking=True).float()

    # Move 'batch_idx' tensor to the same device
    data["batch_idx"] = data["batch_idx"].to(device, non_blocking=True).float()

    # Move 'cls' tensor to the same device
    data["cls"] = data["cls"].to(device, non_blocking=True).float()

    # Move 'onj_cls' integer to the same device
    data["onj_cls"] = torch.tensor(data["onj_cls"], device=device, dtype=torch.int64)

    return data
