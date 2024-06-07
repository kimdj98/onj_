import ultralytics
import torch
import hydra
from omegaconf import DictConfig
from enum import Enum
import cv2
import json
import numpy as np
import os
import wandb


class Modal(Enum):
    CT = "CT"
    PA = "PA"


def predict(model: torch.nn.Module, image: str, label: str = None, output_dir: str = None) -> None:
    results = model.predict(image, conf=0.01, iou=0.3, save=True)
    if label:
        # # read each lines of label.txt file
        # with open(label, "r") as f:
        #     lines = f.readlines()

        # # draw bounding box
        # for line in lines:
        #     line = line.strip().split(" ")
        #     x1, y1, x2, y2 = map(float, line[1:])
        #     img = img.draw_box([x1, y1, x2, y2], color="red", width=3)
        # label is json file
        with open(label, "r") as f:
            label = json.load(f)

        for box in label["bbox"]:
            c_x, c_y, w, h = box["coordinates"]
            c_x, c_y, w, h = c_x * label["width"], c_y * label["height"], w * label["width"], h * label["height"]
            img = cv2.imread(output_dir + "/runs/detect/predict/" + image.split("/")[-1])
            img = np.array(img)
            img = cv2.rectangle(
                img, (int(c_x - w / 2), int(c_y - h / 2)), (int(c_x + w / 2), int(c_y + h / 2)), (255, 0, 0), 5
            )

        cv2.imwrite(output_dir + "/runs/detect/predict/" + image.split("/")[-1], img)


@hydra.main(version_base="1.1", config_path="../../../config", config_name="config")
def train(cfg: DictConfig):
    modal = Modal.PA
    # model = ultralytics.YOLO("/mnt/aix22301/onj/outputs/2024-05-06/15-20-18/runs/detect/train/weights/last.pt")
    model = ultralytics.YOLO("yolov8l.pt")

    if modal == Modal.CT:
        imgsz = cfg.data.CT_dim
    elif modal == Modal.PA:
        imgsz = cfg.data.PA_dim

    dataset_yaml = "/mnt/aix22301/onj/code/data/yolo_dataset.yaml"

    model.train(
        data=dataset_yaml,
        lr0=1e-3,
        lrf=1e-2,
        epochs=2000,
        device="0",
        batch=-1,
        imgsz=list(imgsz),
        scale=0.0,
        augment=False,
        hsv_h=0,
        hsv_s=0,
        hsv_v=0,
        translate=0.0,
        fliplr=0.0,
        crop_fraction=0.0,
        mosaic=0.0,
    )

    # CT prediction
    if modal == Modal.CT:
        results = model.predict(
            "/mnt/aix22301/onj/dataset/v0/YOLO/images/train/EW-0012_20200216_MDCT_axial_60.jpg",
            conf=0.01,
            save=True,
        )

    # PA prediction
    elif modal == Modal.PA:
        results = model.predict(
            "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0015/panorama/20160116.jpg",
            conf=0.015,
            save=True,
        )


@hydra.main(version_base="1.1", config_path="../../../config", config_name="config")
def test(cfg: DictConfig):
    modal = Modal.PA
    model = ultralytics.YOLO(
        # "/mnt/aix22301/onj/code/outputs/2024-03-12/yolo_v8m_epoch50/runs/detect/train/weights/last.pt"
    )
    # CT prediction
    if modal == Modal.CT:
        predict(model, "/mnt/aix22301/onj/dataset/v0/YOLO/images/train/EW-0012_20200216_MDCT_axial_60.jpg")

    # # PA prediction
    # elif modal == Modal.PA:
    #     predict(
    #         model,
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0015/panorama/20160116.jpg",
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0015/panorama/label.json",
    #         output_dir=os.getcwd(),
    #     )

    #     predict(
    #         model,
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0019/panorama/20210506.jpg",
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0019/panorama/label.json",
    #         output_dir=os.getcwd(),
    #     )

    #     predict(
    #         model,
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0055/panorama/20110916.jpg",
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0055/panorama/label.json",
    #         output_dir=os.getcwd(),
    #     )

    #     predict(
    #         model,
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0065/panorama/20160802.jpg",
    #         "/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0065/panorama/label.json",
    #         output_dir=os.getcwd(),
    #     )

    # predict(
    #     model,
    #     "/mnt/aix22301/onj/dataset/v0/CLS_PA/Non_ONJ/EW-0002.jpg",
    # )


if __name__ == "__main__":
    train()
