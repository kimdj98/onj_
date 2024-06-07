import sys

sys.path.append("/mnt/aix22301/onj/code")

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import hydra
import logging

from data.data_dicts import LoadJsonLabeld, SelectSliced, get_data_dicts
from data.utils import Modal, Direction
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
    RandAffined,
    RandFlipd,
    RandGaussianNoised,
    Transposed,
)
from model.backbone.classifier.ResNet3D import resnet18_3d, resnet34_3d, resnet50_3d, resnet101_3d

from model.backbone.classifier.backbone_PA_2D import ClassifierModel, YOLOClassifier
import ultralytics

from model.fusor.concat import *  # import all concat models
from model.backbone.utils import FeatureExpand

# from losses.losses import CrossEntropyLoss # custom loss
from torch.nn import functional as F
import torch.optim as optim

from utils.plotting import plot_auroc
from utils.metrics import BinaryAccuracy, BinaryAUROC
from utils.metrics import CrossEntropyLoss
from utils.plotting import apply_grad_cam, apply_grad_cam_3d
from tqdm import tqdm
import matplotlib.pyplot as plt

from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def visualize(cfg):
    # step1: load data
    CT_dim_x = cfg.data.CT_dim[0]
    CT_dim_y = cfg.data.CT_dim[1]
    PA_dim_x = cfg.data.PA_dim[0]
    PA_dim_y = cfg.data.PA_dim[1]

    transforms = Compose(
        [
            LoadImaged(keys=["CT_image", "PA_image"]),
            EnsureChannelFirstd(keys=["CT_image", "PA_image"]),
            LoadJsonLabeld(keys=["CT_annotation"]),  # Use the custom transform for labels
            ScaleIntensityRanged(
                keys=["CT_image"], a_min=-1000, a_max=2500, b_min=0.0, b_max=1.0, clip=False
            ),  # NOTE: check the range
            # ScaleIntensityRangePercentilesd(
            #     keys=["image"], lower=0, upper=100, b_min=0, b_max=1, clip=False, relative=False
            # ),
            ScaleIntensityRanged(keys=["PA_image"], a_min=0, a_max=255, b_min=0.0, b_max=1.0, clip=False),
            Rotate90d(keys=["CT_image"], spatial_axes=(0, 1)),
            Flipd(keys=["CT_image"], spatial_axis=2),
            SelectSliced(keys=["CT_image", "CT_SOI"]),
            Resized(keys=["CT_image"], spatial_size=(CT_dim_x, CT_dim_y, 64), mode="trilinear"),
            Resized(keys=["PA_image"], spatial_size=(PA_dim_x, PA_dim_y), mode="bilinear"),
            # RandAffined(
            #     mode="bilinear",
            #     keys=["CT_image"],
            #     prob=0.5,
            #     spatial_size=(CT_dim_x, CT_dim_y, 64),
            #     # rotate_range=(0.2, 0.2, 0.2),
            #     translate_range=(10, 10, 10),
            #     # scale_range=(0.1, 0.1, 0.1),
            # ),
            # RandAffined(
            #     mode="bilinear",
            #     keys=["PA_image"],
            #     prob=0.5,
            #     spatial_size=(PA_dim_x, PA_dim_y),
            #     translate_range=(10, 10),
            # ),
            # RandFlipd(keys=["CT_image"], spatial_axis=0, prob=0.5),
            # RandGaussianNoised(keys=["CT_image"], std=0.01, prob=0.5),
            ToTensord(keys=["CT_image"]),
        ]
    )

    # ======================================================================================
    # Create data_dicts -> dataset -> dataloader
    BASE_PATH = Path(cfg.data.data_dir)
    train_data_dicts, val_data_dicts, test_data_dicts = get_data_dicts(
        BASE_PATH, includes=[Modal.CBCT, Modal.MDCT, Modal.PA], random_state=cfg.data.random_state
    )

    train_dataset = Dataset(data=train_data_dicts, transform=transforms)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    val_dataset = Dataset(data=val_data_dicts, transform=transforms)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    test_dataset = Dataset(data=test_data_dicts, transform=transforms)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.train.batch_size, shuffle=True)

    # ======================================================================================
    # Initialize model, loss function, optimizer, metrics
    if cfg.model.CT == "resnet18":
        model_3d = resnet18_3d()

    elif cfg.model.CT == "resnet50":
        model_3d = resnet50_3d()

    elif cfg.model.CT == "unet":
        pass

    if cfg.train.pretrained_CT:
        model_3d.load_state_dict(torch.load(cfg.train.pretrained_CT))
        print(f"Pretrained model {cfg.train.pretrained_CT} loaded")

    if cfg.model.PA == "yolo":
        # if cfg.train.pretrained_PA:
        #     yolo_model = ultralytics.YOLO(cfg.train.pretrained_PA)
        #     print(f"Pretrained model {cfg.train.pretrained_PA} loaded")
        yolo_model = ultralytics.YOLO(cfg.train.pretrained_YOLO)
        classifier_model = ClassifierModel(num_classes=2)
        model_2d = YOLOClassifier(yolo_model, classifier_model)
        model_2d.load_state_dict(torch.load(cfg.train.pretrained_PA))
        print(f"Pretrained model {cfg.train.pretrained_PA} loaded")

    if cfg.model.fusion == "concat":
        # model = ConcatModel1(model_2d, model_3d, input_size=256 * 64 * 32 + 2048, num_classes=2)
        # model = ConcatModel2(model_2d, model_3d, input_size=256 * 64 * 32 + 2048, num_classes=2)
        # model = ConcatModel3(model_2d, model_3d, input_size=32 * 64 * 32 + 2048, num_classes=2)
        # model = ConcatModel4(model_2d, model_3d, input_size=32 * 64 * 32 + 2048, num_classes=2)
        # model = ConcatModel4_dropout(model_2d, model_3d, input_size=32 * 64 * 32 + 2048, num_classes=2)
        model = ConcatModel5(model_2d, model_3d, input_size=4, num_classes=2)
        # model = ConcatModel6(model_2d, model_3d, input_size=1 * 64 * 32 + 2048, num_classes=2)
        # model = ConcatModel7(model_2d, model_3d, input_size=512 + 512, num_classes=2)
        # model = ConcatModel7_dropout(model_2d, model_3d, input_size=512 + 512, num_classes=2)
        # model = ConcatModel8(model_2d, model_3d, input_size=)

    elif cfg.model.fusion == "attention":
        pass

    elif cfg.model.fusion == "mamba":
        pass

    if cfg.model.freeze_2d == True:
        model.freeze_2d()

    if cfg.model.freeze_3d == True:
        model.freeze_3d()

    if cfg.train.pretrained_fusion:
        model.load_state_dict(torch.load(cfg.train.pretrained_fusion))
        print(f"Pretrained model {cfg.train.pretrained_fusion} loaded")

    # switch model and dataset device to cuda
    device = torch.device(f"cuda:{cfg.train.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    model = model.to(device)

    feature_expand_low = FeatureExpand(in_channels=cfg.model.low_channels, out_channels=cfg.model.low_expanded).to(
        device
    )
    feature_expand_middle = FeatureExpand(in_channels=cfg.model.mid_channels, out_channels=cfg.model.mid_expanded).to(
        device
    )
    feature_expand_high = FeatureExpand(in_channels=cfg.model.high_channels, out_channels=cfg.model.high_expanded).to(
        device
    )

    # Create results.txt
    with open("results.txt", "w") as f:
        f.write("Results\n")

    best_AUROC = 0.0
    best_ACC = 0.0
    # training loop
    for epoch in range(cfg.train.epoch):
        model.train()
        running_loss = 0.0

        train_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch}")
        val_iterator = tqdm(val_dataloader, desc=f"Validating")
        test_iterator = tqdm(test_dataloader, desc=f"Testing")
        for i, batch in enumerate(test_iterator):
            if cfg.mode != "train":  # block training when cfg.mode is not "train"
                logging.warning("Training is skipped")
                viz_2d = apply_grad_cam(
                    model,
                    [
                        model.model_2d.classifier_model.conv1,
                        model.model_2d.classifier_model.conv2,
                        model.model_2d.classifier_model.conv3,
                        model.model_2d.classifier_model.conv4,
                    ],
                    input_tensor=batch,
                )
                cv2.imwrite(f"viz_{i}HM_label_{batch['PA_label'].item()}.jpg", viz_2d[0].swapaxes(0, 1))
                cv2.imwrite(
                    f"viz_{i}PA_label_{batch['PA_label'].item()}.jpg",
                    np.array(batch["PA_image"][0].permute(2, 1, 0).cpu() * 255),
                )
                # viz_3d = apply_grad_cam_3d(model, [model.model_3d.layer4], input_tensor=batch, slice_idx=32)
                # cv2.imwrite(f"viz_{i}HM_label_{batch['PA_label'].item()}.jpg", viz_3d[0].swapaxes(0, 1))
                # cv2.imwrite(
                #     f"viz_{i}PA_label_{batch['PA_label'].item()}.jpg",
                #     np.array(batch["PA_image"][0].permute(2, 1, 0).cpu() * 255),
                # )
                continue
            assert (
                batch["CT_label"][0] == batch["PA_label"][0]
            )  # check if the labels are the same (if assertion fails, issue in data preprocessing!)
            # switch device

        # validation loop
        with torch.no_grad():
            true_labels = []
            predictions = []

            val_iterator = tqdm(val_dataloader, desc=f"Validating")
            model.eval()
            running_loss = 0.0
            for i, batch in enumerate(val_iterator):
                pass


if __name__ == "__main__":
    visualize()
