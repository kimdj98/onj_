# initialize_labels.py
# Description: This file initializes the labels non_ONJ folder

import sys
import os
from pathlib import Path
import hydra
import json
from omegaconf import DictConfig

def initialize(base_dir:Path , label:str="Non_ONJ", modal:str="CBCT" ,direction:str="axial"):
    # iterate through each patient folder
    for patient in (base_dir / label).glob("*"):
    # check if patient folder has the modal folder
        if not (patient / modal).exists():
            continue

        modal_path = (patient / modal)

        if not (modal_path).exists():
            continue

        modal_path = list(modal_path.glob("*/" + modal + "_" + direction))[0]

        # check if label.json already exists
        if (modal_path / "label.json").exists():
            continue

        # create the label.json file
        label_data = {"SOI": [0, 0]}
        with open(modal_path / "label.json", "w") as f:
            json.dump(label_data, f)
            pass


    # check if folder exists
    # check if label already exists
    # if label not exists create the label and initialize the SOI to [zero, zero]
    # write the SOI (initialized to zero) JSON data to each patient folder


@hydra.main(version_base="1.1", config_path="../config", config_name="config") # version_base 1.1: changes working directory to outputs folder
def main(cfg: DictConfig):
    base_dir = Path(cfg.data.data_dir) 
    initialize(base_dir, label="Non_ONJ", modal="CBCT", direction="axial")



if __name__ == "__main__":
    main()
