import yaml
from pathlib import Path

import os
import hydra
from omegaconf import DictConfig


@hydra.main(version_base="v1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    def inner(split: str = "train"):
        base_dir = Path(cfg.data.data_dir)
        # read train.txt
        with open(f"{base_dir}/{split}.txt", "r") as f:
            patients = f.readlines()
            patients = [line.split(" ")[0].split("/")[1] for line in patients]

        for patient in patients:
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
            os.system(f"cp {patient_dir}/panorama/*.jpg {base_dir}/YOLO_PA/images/{split}/{patient_dir.name}.jpg")

            # move label.json to YOLO_PA_train/labels
            os.system(f"cp {patient_dir}/panorama/label.json {base_dir}/YOLO_PA/labels/{split}/{patient_dir.name}.json")

    inner("train")
    print("Create YOLO_PA_train done")

    inner("val")
    print("Create YOLO_PA_val done")

    inner("test")
    print("Create YOLO_PA_test done")


if __name__ == "__main__":
    main()
