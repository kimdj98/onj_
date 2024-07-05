# yolo_cls_pa_dataset.py: This script is used to create a dataset for YOLO_CLS_PA_train, YOLO_CLS_PA_val, and YOLO_CLS_PA_test.
# The dataset is created by copying the images and label.json files from the ONJ_labeling folder to the YOLO_PA folder.
# The script reads the train.txt, val.txt, and test.txt files to get the list of patients and then copies the images and label.json files to the respective folders in the YOLO_PA folder.

import yaml
from pathlib import Path

import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="v1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    def inner(split: str = "train"):
        base_dir = Path(cfg.data.data_dir)
        BASE_PATH = f"/mnt/aix22301/onj/dataset/v0"
        DATA_PATH = f"YOLO_CLS_PA"
        if not os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/train/ONJ")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/train/non_ONJ")

            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/val/ONJ")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/val/non_ONJ")

            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/test/ONJ")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/test/non_ONJ")

        # read train.txt
        with open(f"{base_dir}/{split}.txt", "r") as f:
            patients = f.readlines()
            labels = [line.split(" ")[1] for line in patients]  # 0 or 1
            patients = [line.split(" ")[0].split("/")[1] for line in patients]  # patient name (EW-0001)

        for i in range(len(patients)):
            patient = patients[i]
            label = int(labels[i][0])

            if label == 1:
                patient_dir = base_dir / "ONJ_labeling" / patient
                # if patient_dir has no panorama folder, skip
                if not (patient_dir / "panorama").exists():
                    continue

                # if patient_dir has no images, skip
                if len(os.listdir(patient_dir / "panorama")) == 0:
                    continue

                # move image to YOLO_CLS_PA_train/images
                os.system(f"cp {patient_dir}/panorama/*.jpg {base_dir}/{DATA_PATH}/{split}/ONJ/{patient_dir.name}.jpg")

            elif label == 0:
                patient_dir = base_dir / "Non_ONJ_soi" / patient
                # if patient_dir has no panorama folder, skip
                if not (patient_dir / "panorama").exists():
                    continue

                # if patient_dir has no images, skip
                if len(os.listdir(patient_dir / "panorama")) == 0:
                    continue

                # move image to YOLO_CLS_PA_train/images
                os.system(
                    f"cp {patient_dir}/panorama/*.jpg {base_dir}/{DATA_PATH}/{split}/non_ONJ/{patient_dir.name}.jpg"
                )

    inner("train")
    print("Create YOLO_CLS_PA_train done")

    inner("val")
    print("Create YOLO_CLS_PA_val done")

    inner("test")
    print("Create YOLO_CLS_PA_test done")


if __name__ == "__main__":
    main()
