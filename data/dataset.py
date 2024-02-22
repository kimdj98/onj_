# dataset.py
# TODO: Add the code to load the data from the split and create a dictionary of the data
import hydra
import torch 
import json
import os
from pathlib import Path
from omegaconf import DictConfig

def load_data_dict(conf: DictConfig, modal:str, dir:str, type:str) -> dict:
    data_dicts = []
    DATA_PATH = Path(conf.data.data_dir)
    for patient in DATA_PATH.glob(f"{type}/*"):
        if (patient / modal).exists():
            for modal_dir in (patient / modal).glob("*/*"):
                if not modal_dir.is_dir():
                    continue
                if "nifti" in os.listdir(modal_dir) and "label.json" in os.listdir(modal_dir) and dir in modal_dir.name:
                    data_dicts.append(
                        {
                            "image": modal_dir / "nifti" / "output.nii.gz",
                            "label": modal_dir / "label.json",
                        }
                    )
    return data_dicts

@hydra.main(config_path="../config", config_name="config")
def main(conf: DictConfig):
    data_dict_MDCT_axial = load_data_dict(conf, modal="MDCT", dir="axial", type="ONJ")
    data_dict_CBCT_axial = load_data_dict(conf, modal="CBCT", dir="axial", type="ONJ")
    data_dict_MDCT_coronal = load_data_dict(conf, modal="MDCT", dir="coronal", type="ONJ")
    data_dict_CBCT_coronal = load_data_dict(conf, modal="CBCT", dir="coronal", type="ONJ")

if __name__ == "__main__":
    main()