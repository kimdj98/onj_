import numpy as np
import json
import os

from pathlib import Path

from omegaconf import DictConfig
import hydra
from enum import Enum


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    data_dir = Path(cfg.data.data_dir)
    non_onj_dir = data_dir / "non_onj"

    # check if the patient_label has CT data
    for patient in non_onj_dir.glob("*"):
        has_CT = False
        if list(patient.glob("CBCT")) != []:
            has_CT = True
        elif list(patient.glob("MDCT")) != []:
            has_CT = True

        if has_CT:
            has_CT_label = False
            if list(patient.glob("CBCT/*/CBCT_axial/label.json")) != []:
                has_CT_label = True

            elif list(patient.glob("MDCT/*/MDCT_axial/label.json")) != []:
                has_CT_label = True

            if has_CT_label == False:
                print(f"{patient.name} has no CT label")


if __name__ == "__main__":
    main()
