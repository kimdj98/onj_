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

    # NOTE: switch below two code lines to convert ONJ or Non_ONJ_soi
    MODAL_PATH = DATAPATH / "onj"
    # MODAL_PATH = DATAPATH / "non_onj"

    dicom_to_nifti(MODAL_PATH)

    for f in error_folders:
        print(f)


if __name__ == "__main__":
    main()

    # dicom2nifti.dicom_series_to_nifti(
    #     str("/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0323/MDCT/20211228/MDCT_axial/dicom"),
    #     str("/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0323/MDCT/20211228/MDCT_axial/nifti/output.nii.gz"),
    # )

    # dicom2nifti.dicom_series_to_nifti(
    #     str("/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0323/MDCT/20211228/MDCT_coronal/dicom"),
    #     str("/mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0323/MDCT/20211228/MDCT_coronal/nifti/output.nii.gz"),
    # )


"""
Errors
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/dicom
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0135/CBCT/20220216/CBCT_sagittal/dicom # ignore sagittal
"""
