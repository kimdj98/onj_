# yolo_pa_dataset.py: This script is used to create a dataset for YOLO_PA_train, YOLO_PA_val, and YOLO_PA_test.
# The dataset is created by copying the images and label.json files from the ONJ_labeling folder to the YOLO_PA folder.
# The script reads the train.txt, val.txt, and test.txt files to get the list of patients and then copies the images and label.json files to the respective folders in the YOLO_PA folder.

import yaml
from pathlib import Path

import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="v1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    def inner(split: str = "train"):
        split_ver = "split_v1"
        base_dir = Path(cfg.data.data_dir)
        BASE_PATH = cfg.data.data_dir
        DATA_PATH = f"YOLO_PA2"
        if not os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/images/test")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/train")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/val")
            os.makedirs(f"{BASE_PATH}/{DATA_PATH}/labels/test")

        # read train.txt
        with open(f"{base_dir}/{split_ver}/{split}.txt", "r") as f:
            patients = f.readlines()
            label = [line.split(" ")[1].strip() for line in patients]
            patients = [line.split(" ")[0].split("/")[1] for line in patients]

        for i, patient in enumerate(patients):
            if label[i] == "1":
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
                os.system(
                    f"cp {patient_dir}/panorama/*.jpg {base_dir}/{DATA_PATH}/images/{split}/{patient_dir.name}.jpg"
                )

                # move label.json to YOLO_PA_train/labels
                os.system(
                    f"cp {patient_dir}/panorama/label.json {base_dir}/{DATA_PATH}/labels/{split}/{patient_dir.name}.json"
                )

            elif label[i] == "0":
                patient_dir = base_dir / "Non_ONJ_soi" / patient
                # if patient_dir has no panorama folder, skip
                if not (patient_dir / "panorama").exists():
                    continue

                # # if patient_dir has no label.json, skip
                # if not (patient_dir / "panorama" / "label.json").exists():
                #     continue

                # if patient_dir has no images, skip
                if len(os.listdir(patient_dir / "panorama")) == 0:
                    continue

                # move image to YOLO_PA_train/images
                os.system(
                    f"cp {patient_dir}/panorama/*.jpg {base_dir}/{DATA_PATH}/images/{split}/{patient_dir.name}.jpg"
                )

                # move label.json to YOLO_PA_train/labels
                if (patient_dir / "panorama" / "label.json").exists():
                    # move label.json
                    os.system(
                        f"cp {patient_dir}/panorama/label.json {base_dir}/{DATA_PATH}/labels/{split}/{patient_dir.name}.json"
                    )

                else:
                    # write empty json file
                    with open(f"{base_dir}/{DATA_PATH}/labels/{split}/{patient_dir.name}.json", "w") as f:
                        f.write("{}")

    inner("train")
    print("Create YOLO_PA_train done")

    inner("val")
    print("Create YOLO_PA_val done")

    inner("test")
    print("Create YOLO_PA_test done")


if __name__ == "__main__":
    main()
