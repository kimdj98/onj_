from pathlib import Path
import os
import hydra
from omegaconf import DictConfig


def find():
    BASE_DIR = Path("/mnt/aix22301/onj/dataset/v2")
    DATA_PATH = BASE_DIR / "YOLO_PA2"
    ONJ_PATH = BASE_DIR / "onj"
    NON_ONJ_PATH = BASE_DIR / "non_onj"

    with open(BASE_DIR / "patient_list.txt", "w") as f:
        # just write empty file and then append patient file dir
        pass

    for split in (DATA_PATH / "images").iterdir():
        for img in split.iterdir():
            patient = img.name.split(".")[0]
            has_CT = False
            has_PA = False

            if (ONJ_PATH / patient).exists():
                if (ONJ_PATH / patient / "panorama").exists():
                    has_PA = True

                if (ONJ_PATH / patient / "CBCT").exists():
                    has_CT = True

                if (ONJ_PATH / patient / "MDCT").exists():
                    has_CT = True

                if not has_CT or not has_PA:
                    print(f"ONJ: {patient} has CT: {has_CT}, PA: {has_PA}")

                    # write to patient_list.txt
                    with open(BASE_DIR / "patient_list.txt", "a") as f:
                        f.write(f"{img}\n")

            elif (NON_ONJ_PATH / patient).exists():
                if (NON_ONJ_PATH / patient / "panorama").exists():
                    has_PA = True

                if (NON_ONJ_PATH / patient / "CBCT").exists():
                    has_CT = True

                if (NON_ONJ_PATH / patient / "MDCT").exists():
                    has_CT = True

                if not has_CT or not has_PA:
                    print(f"NON ONJ: {patient} has CT: {has_CT}, PA: {has_PA}")

                    # write to patient_list.txt
                    with open(BASE_DIR / "patient_list.txt", "a") as f:
                        f.write(f"{img}\n")


if __name__ == "__main__":
    find()
