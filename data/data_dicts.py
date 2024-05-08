import sys

sys.path.append("/mnt/4TB1/onj/onj_project")

import torch
import json

from sklearn.model_selection import train_test_split
from pathlib import Path

import hydra
from omegaconf import DictConfig

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
from data.utils import Modal, Direction
from monai.config.type_definitions import NdarrayTensor

# from tqdm import tqdm


def add_CT_data_dicts(
    CT_modal: Modal, data_dicts: list, PATIENT_PATH: Path, ONJ: bool = True, exists: bool = True, CT_dir="axial"
):
    if exists:
        modal_dir = PATIENT_PATH / CT_modal.value
        modal_dir = list(modal_dir.glob(f"*/{CT_modal.value}_{CT_dir}"))[0]

        annotations = json.load(open(modal_dir / "label.json", "r"))

        if "SOI" in annotations.keys():
            SOI = annotations["SOI"]
        else:
            SOI = [0, 0]

        data_dicts.append(
            {
                "CT_image": modal_dir / "nifti" / "output.nii.gz",
                "CT_annotation": modal_dir / "label.json",
                "CT_SOI": SOI,
                "CT_label": ONJ,
            }
        )


def add_panorama_data_dicts(data_dicts: list, PATIENT_PATH: Path, ONJ: bool = True, exists: bool = True):
    pass


def add_BoneSPECT_data_dicts(data_dicts: list, PATIENT_PATH: Path, ONJ: bool = True, exists: bool = True):
    pass


def add_ClinicalData_data_dicts(data_dicts: list, PATIENT_PATH: Path, ONJ: bool = True, exists: bool = True):
    pass


def get_data_dicts(
    BASE_PATH: Path, includes: list[Modal], split_ratio: list = [0.80, 0.19, 0.01], random_state: int = 42
):
    ONJ_PATH = BASE_PATH / "ONJ_labeling"
    NON_ONJ_PATH = BASE_PATH / "Non_ONJ_soi"

    data_dicts = []
    patients = list(ONJ_PATH.glob("*")) + list(NON_ONJ_PATH.glob("*"))
    labels = [1] * len(list(ONJ_PATH.glob("*"))) + [0] * len(list(NON_ONJ_PATH.glob("*")))
    random_state = random_state
    # train val test split
    # random_state is the seed used by the random number generator

    patients_train, patients_test, labels_train, labels_test = train_test_split(
        patients, labels, test_size=split_ratio[2], random_state=random_state
    )
    patients_train, patients_val, labels_train, labels_val = train_test_split(
        patients_train, labels_train, test_size=split_ratio[1], random_state=random_state
    )

    def helper(data_dicts: list, patients: list, labels: list, includes: list):
        for patient, label in zip(patients, labels):
            mdct_exists = (patient / "MDCT").is_dir()
            cbct_exists = (patient / "CBCT").is_dir()
            panorama_exists = (patient / "panorama").is_dir()
            bonespect_exists = (patient / "BoneSPECT").is_dir()
            clinicaldata_exists = (patient / "ClinicalData").is_dir()

            if Modal.MDCT in includes:
                add_CT_data_dicts(Modal.MDCT, data_dicts, patient, label, mdct_exists)
            # if not mdct_exists or not cbct_exists: # when MDCT and CBCT both exists in the patient use only MDCT
            if Modal.CBCT in includes:
                add_CT_data_dicts(Modal.CBCT, data_dicts, patient, label, cbct_exists)
            if Modal.panorama in includes:
                add_panorama_data_dicts(data_dicts, patient, label, panorama_exists)
            if Modal.BoneSPECT in includes:
                add_BoneSPECT_data_dicts(data_dicts, patient, label, bonespect_exists)
            if Modal.ClinicalData in includes:
                add_ClinicalData_data_dicts(data_dicts, patient, label, clinicaldata_exists)

    train_data_dicts = []
    val_data_dicts = []
    test_data_dicts = []

    helper(train_data_dicts, patients_train, labels_train, includes)
    helper(val_data_dicts, patients_val, labels_val, includes)
    helper(test_data_dicts, patients_test, labels_test, includes)

    return train_data_dicts, val_data_dicts, test_data_dicts


class LoadJsonLabeld(MapTransform):
    """
    Custom transform to load bounding box coordinates from a JSON file if the data has label.json.
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False, x_dim: int = 512, y_dim: int = 512):
        MapTransform.__init__(self, keys, allow_missing_keys)
        self.x_dim = x_dim
        self.y_dim = y_dim

    def __call__(self, data: dict) -> dict:
        total_slices = data["CT_image"].shape[-1]

        t = torch.zeros((total_slices, 5))
        data_path = data["CT_annotation"]

        if data_path == "":
            data["CT_annotation"] = t
            return data

        with open(data_path, "r") as file:
            annotations = json.load(file)

        # when labels have key slices
        if "slices" in annotations.keys():
            num_labels = len(annotations["slices"])
            label_start = annotations["slices"][0]["slice_number"]
            label_end = annotations["slices"][-1]["slice_number"]

            t = torch.zeros((total_slices, 5), dtype=torch.float32)

            for i in range(num_labels):
                slice = annotations["slices"][i]
                slice_number = slice["slice_number"]
                # TODO: Check if the coordinate type is x, y, w, h
                x = slice["bbox"][0]["coordinates"][0]
                y = slice["bbox"][0]["coordinates"][1]
                w = slice["bbox"][0]["coordinates"][2]
                h = slice["bbox"][0]["coordinates"][3]

                t[slice_number] = torch.tensor([1.0, x, y, w, h])

        if data["CT_SOI"] != [0, 0]:
            SOI = data["CT_SOI"]
            t = t[SOI[0] : SOI[1]]

        data["CT_annotation"] = t

        return data


class SelectSliced(MapTransform):
    """
    Custom transform to select slices we are interested from a 3D image
    """

    def __init__(self, keys: str, allow_missing_keys: bool = False):
        MapTransform.__init__(self, keys, allow_missing_keys)

    def __call__(self, data: dict) -> dict:
        if data["CT_SOI"] == [0, 0]:  # SOI not found
            return data

        else:
            SOI = data["CT_SOI"]
            data["CT_image"] = data["CT_image"][..., SOI[0] : SOI[1]]
            return data


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    CT_dim_x = cfg.data.CT_dim[0]
    CT_dim_y = cfg.data.CT_dim[1]

    transforms = Compose(
        [
            LoadImaged(keys=["CT_image"]),
            EnsureChannelFirstd(keys=["CT_image"]),
            LoadJsonLabeld(keys=["CT_annotation"]),  # Use the custom transform for labels
            ScaleIntensityRanged(keys=["CT_image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=True),
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
            # ),
            Rotate90d(keys=["CT_image"], spatial_axes=(0, 1)),
            Flipd(keys=["CT_image"], spatial_axis=2),
            SelectSliced(keys=["CT_image", "CT_SOI"]),
            Resized(keys=["CT_image"], spatial_size=(CT_dim_x, CT_dim_y, 64), mode="trilinear"),
            ToTensord(keys=["CT_image"]),
        ]
    )

    # Create data_dicts
    BASE_PATH = Path(cfg.data.data_dir)
    train_data_dicts, val_data_dicts, test_data_dicts = get_data_dicts(BASE_PATH, includes=[Modal.CBCT, Modal.MDCT])

    # train:val:test = 299:38:41 (example)

    # Create a MONAI dataset
    dataset = Dataset(data=train_data_dicts, transform=transforms)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    print(f"Total patients: {dataset.__len__()}")

    # import 3d model
    # 1. load model 3d resnet
    # model = torch.hub.load('facebookresearch/pytorchvideo', 'slow_r50', pretrained=False)
    from model.backbone.classifier.ResNet3D import resnet18_3d

    model = resnet18_3d()
    # convert model weights from cpu to gpu
    model = model.to(device=torch.device("cuda:0"))
    # model = model.cuda()
    from torchsummary import summary

    summary(model, input_size=(1, 64, 512, 512))

    pass

    for i, data in enumerate(dataloader):
        print(data["CT_image"].shape)
        print(data["CT_annotation"].shape)
        print(data["CT_SOI"])
        print(data["CT_label"])
        print("=====================================")
        if i == 10:
            break


if __name__ == "__main__":
    main()
