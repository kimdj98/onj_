# File Description: Check if output.nii.gz exists in nifti folder for each patient
import os
import sys
import hydra
from omegaconf import DictConfig
from pathlib import Path

error_folders = []
CT_MODALS = ["MDCT", "CBCT"]


def check_niftis(data_path):
    for patient in list(data_path.glob("*")):
        for modal in list(patient.glob("*")):
            if modal.name in CT_MODALS:
                modals = list(modal.glob("*/*"))
                for modal in modals:
                    try:
                        # check if output.nii.gz exists in nifti folder
                        if "nifti" in os.listdir(modal):
                            if "output.nii.gz" in os.listdir(modal / "nifti"):
                                # print("Checked", modal / "nifti" / "output.nii.gz")
                                pass
                            else:
                                print("Error in checking", modal / "nifti" / "output.nii.gz")
                                error_folders.append(modal)
                    except:
                        print("Error in checking", modal)
                        continue


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    DATAPATH = Path(cfg.data.data_dir)

    # NOTE: switch below two code lines to check ONJ or Non_ONJ_soi
    MODAL_PATH = DATAPATH / "onj"
    check_niftis(MODAL_PATH)

    MODAL_PATH = DATAPATH / "non_onj"
    check_niftis(MODAL_PATH)


if __name__ == "__main__":
    main()

    for f in error_folders:
        print(f)
