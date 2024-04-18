import os
import sys
import hydra
from omegaconf import DictConfig

from pathlib import Path
import dicom2nifti
from tqdm import tqdm

error_folders = []
CT_MODALS = ["MDCT", "CBCT"]


# Convert ONJ dicom file to nifti
def dicom_to_nifti(data_path):
    for patient in tqdm(list(data_path.glob("*"))):
        for modal in list(patient.glob("*")):
            if modal.name in CT_MODALS:
                modals = list(modal.glob("*/*"))
                for modal in modals:
                    try:
                        if "dcm" in os.listdir(modal):  # check if dicom folder named 'dcm' exists
                            os.rename(modal / "dcm", modal / "dicom")

                        os.makedirs(modal / "nifti", exist_ok=True)
                        print("Converting", modal / "dicom")

                        dicom2nifti.dicom_series_to_nifti(str(modal / "dicom"), str(modal / "nifti" / "output.nii.gz"))

                        print("Converted", str(modal / "nifti" / "output.nii.gz"))

                    except:
                        print("Error in converting", modal / "dicom")
                        error_folders.append(modal / "dicom")
                        continue


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    DATAPATH = Path(cfg.data.data_dir)
    # ONJ_PATH = DATAPATH / "ONJ_labeling"
    NONONJ_PATH = DATAPATH / "Non_ONJ_soi"

    # dicom_to_nifti(ONJ_PATH)
    dicom_to_nifti(NONONJ_PATH)

    for f in error_folders:
        print(f)


if __name__ == "__main__":
    # main()

    dicom2nifti.dicom_series_to_nifti(
        str("/mnt/4TB1/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom"),
        str("/mnt/4TB1/onj/dataset/v0/Non_ONJ/EW-0429/CBCT/20200616/CBCT_axial/nifti/output.nii.gz"),
    )


"""
Errors
/mnt/4TB1/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom
/mnt/4TB1/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/dicom
/mnt/4TB1/onj/dataset/v0/Non_ONJ_soi/EW-0135/CBCT/20220216/CBCT_sagittal/dicom # ignore sagittal
"""
