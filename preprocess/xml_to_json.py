# xml_to_json.py
# Description: This file initializes the labels non_ONJ_SOI folder convert SOIs in .xml to json file

import re
from pathlib import Path
import hydra
import json
from omegaconf import DictConfig

def initialize(base_dir:Path , label:str="Non_ONJ_soi", modal:str="MDCT" ,direction:str="axial"):
    # iterate through each patient folder
    for patient in (base_dir / label).glob("*"):
        # check if patient folder has the modal folder
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


def xml_to_json(base_dir:Path , label:str="Non_ONJ_soi", modal:str="MDCT" ,direction:str="axial"):
    def helper(xml_file: Path):
        return int(re.findall(r'\d+', (xml_file).name.split(".")[0])[0])

    # iterate through each patient folder
    for patient in (base_dir / label).glob("*"):
        # check if patient folder has the modal folder
        modal_path = (patient / modal)
        if not (modal_path).exists():
            continue

        modal_path = list(modal_path.glob("*/" + modal + "_" + direction))[0]

        # # check if label.json already exists
        # if (modal_path / "label.json").exists():
        #     continue

        # Search for xml files in jpg folder
        xml_files = list(modal_path.glob("jpg/*.xml"))
        if len(xml_files) == 0:
            continue

        assert len(xml_files) == 2, f"Patient {patient.name}, Modal {modal}, Direction {direction} has {len(xml_files)} xml files"
        pass
        
        SOI = [helper(xml_file) for xml_file in xml_files]
        SOI.sort()
        SOI = {"SOI": SOI}

        # create the label.json file
        with open(modal_path / "label.json", "w") as f:
            json.dump(SOI, f)
            pass


    # check if folder exists
    # check if label already exists
    # if label not exists create the label and initialize the SOI to [zero, zero]
    # write the SOI (initialized to zero) JSON data to each patient folder

# 1. initialize -> initialize the labels non_ONJ_SOI folder labels.json to SOI: [0, 0]
# 2. create_labels -> fill in the SOI:[0,0] if it has xml files inside jpg folder

@hydra.main(version_base="1.1", config_path="../config", config_name="config") # version_base 1.1: changes working directory to outputs folder
def main(cfg: DictConfig):
    base_dir = Path(cfg.data.data_dir)
    initialize(base_dir, label="Non_ONJ_soi", modal="CBCT", direction="axial")
    xml_to_json(base_dir, label="Non_ONJ_soi", modal="CBCT", direction="axial")
    initialize(base_dir, label="Non_ONJ_soi", modal="CBCT", direction="coronal")
    xml_to_json(base_dir, label="Non_ONJ_soi", modal="CBCT", direction="coronal")
    initialize(base_dir, label="Non_ONJ_soi", modal="MDCT", direction="axial")
    xml_to_json(base_dir, label="Non_ONJ_soi", modal="MDCT", direction="axial")
    initialize(base_dir, label="Non_ONJ_soi", modal="MDCT", direction="coronal")
    xml_to_json(base_dir, label="Non_ONJ_soi", modal="MDCT", direction="coronal")


if __name__ == "__main__":
    main()



