import os
import sys
import hydra
from omegaconf import DictConfig

from pathlib import Path
import SimpleITK as sitk
from tqdm import tqdm
import cv2
import re

error_folders = []
CT_MODALS = ["MDCT", "CBCT"]

def rename_files(files: list[str], parent: Path, modal: str):
    for file in files:
        # extract number in file.name file.name is like "CB1.jpg"
        # find it using regular expression
        num = re.findall(r"\d+", file.name)[0]
        new_num = str(int(num) - 1)
        new_name = file.name.replace(num, new_num)
        os.rename(file, parent / modal / new_name)


def preprocess(data_path, patients):
    """
    Delete the first file where there is slice info in the first file of the patient folder
    """
    # first assert that the first image in jpg folder and the second image in jpg folder are not the same size
    for patient in tqdm(list(data_path.glob("*"))):
        if patient.name in patients: # only for the patients in the list: may need to erase this to automatically process all patients
            for modal in list(patient.glob("*")):
                if modal.name in CT_MODALS:
                    modals = list(modal.glob("*/*"))
                    for modal in modals:
                        try:
                            if "jpg" in os.listdir(modal):
                                jpg_files = list(modal.glob("jpg/*"))
                                if len(jpg_files) > 1:
                                    # compare the size of the jpg file
                                    jpg_files = sorted(jpg_files)
                                    if cv2.imread(str(jpg_files[0])).shape != cv2.imread(str(jpg_files[1])).shape:                                         
                                        # check if the first and second image are the same size
                                        # if they are the same size, no slice info
                                        # if they are different size, slice info at the first image delete the first image
                                        os.remove(jpg_files[0])
                                        print("Deleted", jpg_files[0])

                                        # also delete the first file in dicom folder
                                        if "dicom" in os.listdir(modal):
                                            dcm_files = list(modal.glob("dicom/*"))
                                            dcm_files = sorted(dcm_files)
                                            if len(dcm_files) > 1:
                                                os.remove(dcm_files[0])
                                                print("Deleted", dcm_files[0])
                                        
                                        # convert each image CB1 -> CB0, CB2 -> CB1, CB3 -> CB2 ...
                                        rename_files(jpg_files, modal, "jpg")
                                        # convert each dicom CB1 -> CB0, CB2 -> CB1, CB3 -> CB2 ...
                                        rename_files(dcm_files, modal, "dicom")

                                        # for file in jpg_files[1:]:
                                        #     # extract number in file.name file.name is like "CB1.jpg"
                                        #     # find it using regular expression
                                        #     num = re.findall(r"\d+", file.name)[0]
                                        #     new_num = str(int(num) - 1)
                                        #     new_name = file.name.replace(num, new_num)
                                        #     os.rename(file, modal / "jpg" / new_name)

                                        # for file in dcm_files[1:]:
                                        #     num = re.findall(r"\d+", file.name)[0]
                                        #     new_num = str(int(num) - 1)
                                        #     new_name = file.name.replace(num, new_num)
                                        #     os.rename(file, modal / "dicom" / new_name)

                        except:
                            print("Error in deleting", modal / "jpg")
                            error_folders.append(modal / "jpg")
                            continue


def convert_modal(modal_folder:Path):
    # convert the modal folder dicom series to nifti output
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(str(modal_folder / "dicom"))
    reader.SetFileNames(dicom_names)
    image = reader.Execute()
    sitk.WriteImage(image, str(modal_folder / "nifti" / "output.nii.gz"))
            
def convert_patient(patient_folder:Path):
    for modal in list(patient_folder.glob("*")):
        if modal.name in CT_MODALS:
            modals = list(modal.glob("*/*"))
            for modal in modals: # for each modal convert dicom to nifti
                try:
                    if "dcm" in os.listdir(modal):  # check if dicom folder named 'dcm' exists
                        os.rename(modal / "dcm", modal / "dicom")

                    os.makedirs(modal / "nifti", exist_ok=True)

                    print("Converting", modal / "dicom")
                    convert_modal(modal)
                    print("Converted", str(modal / "nifti" / "output.nii.gz"))

                except:
                    print("Error in converting", modal / "dicom")
                    error_folders.append(modal / "dicom")
                    continue

# Convert ONJ dicom file to nifti using SimpleITK
def dicom_to_nifti(data_path):
    for patient in tqdm(list(data_path.glob("*"))):
        convert_patient(patient)


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(cfg: DictConfig):
    DATAPATH = Path(cfg.data.data_dir)
    
    # run preprocess code once for each folder
    # below code deletes data with slice info in the first file
    preprocess(DATAPATH / "Non_ONJ_soi", ["EW-0302", "EW-0544", "EW-0543"])
    # preprocess(DATAPATH / "Non_ONJ_soi", ["EW-0302"])
    preprocess(DATAPATH / "ONJ_labeling", ["EW-0478", "EW-0048", "EW-0465", "EW-0050", "EW-0480", "EW-0471", "EW-0533", "EW-0474", "EW-0476", "EW-0069"])

    # NOTE: switch below two code lines to convert ONJ or Non_ONJ_soi
    # MODAL_PATH = DATAPATH / "ONJ_labeling"
    MODAL_PATH = DATAPATH / "Non_ONJ_soi"

    # /mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0302/CBCT/20201022
    convert_modal(MODAL_PATH / "EW-0302" / "CBCT" / "20201022" / "CBCT_axial")
    convert_modal(MODAL_PATH / "EW-0302" / "CBCT" / "20201022" / "CBCT_coronal")
    convert_modal(MODAL_PATH / "EW-0302" / "CBCT" / "20201022" / "CBCT_sagittal")

    MODAL_PATH = DATAPATH / "ONJ_labeling"

    # /mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0069/CBCT/20160817
    convert_modal(MODAL_PATH / "EW-0069" / "CBCT" / "20160817" / "CBCT_axial")
    convert_modal(MODAL_PATH / "EW-0069" / "CBCT" / "20160817" / "CBCT_coronal")
    convert_modal(MODAL_PATH / "EW-0069" / "CBCT" / "20160817" / "CBCT_sagittal")

    # /mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0474/CBCT/20121203
    convert_modal(MODAL_PATH / "EW-0474" / "CBCT" / "20121203" / "CBCT_axial")
    convert_modal(MODAL_PATH / "EW-0474" / "CBCT" / "20121203" / "CBCT_coronal")
    convert_modal(MODAL_PATH / "EW-0474" / "CBCT" / "20121203" / "CBCT_sagittal")

    # /mnt/aix22301/onj/dataset/v0/ONJ_labeling/EW-0476/CBCT/20120808
    convert_modal(MODAL_PATH / "EW-0476" / "CBCT" / "20120808" / "CBCT_axial")
    convert_modal(MODAL_PATH / "EW-0476" / "CBCT" / "20120808" / "CBCT_coronal")
    convert_modal(MODAL_PATH / "EW-0476" / "CBCT" / "20120808" / "CBCT_sagittal")


    # dicom_to_nifti(MODAL_PATH)



    for f in error_folders:
        print(f)


if __name__ == "__main__":
    main()
    

















    # try: 
    #     reader = sitk.ImageSeriesReader()
    #     dicom_names = reader.GetGDCMSeriesFileNames(str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom"))
    #     reader.SetFileNames(dicom_names)
    #     image = reader.Execute()
    #     sitk.WriteImage(image, str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/nifti/output.nii.gz"))
    # except:
    #     error_folders.append("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom")

    # # EW-0068
    # try:
    #     reader = sitk.ImageSeriesReader()
    #     dicom_names = reader.GetGDCMSeriesFileNames(str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/dicom"))
    #     reader.SetFileNames(dicom_names)
    #     image = reader.Execute()
    #     sitk.WriteImage(image, str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/nifti/output.nii.gz"))
    # except:
    #     error_folders.append("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/dicom")
    
    # EW-0544
    # try:
    # reader = sitk.ImageSeriesReader()
    # dicom_names = reader.GetGDCMSeriesFileNames(str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0544/CBCT/20230510/CBCT_axial/dicom"))
    # reader.SetFileNames(dicom_names)
    # image = reader.Execute()
    # sitk.WriteImage(image, str("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0544/CBCT/20230510/CBCT_axial/nifti/output.nii.gz"))
    # except:
    #     error_folders.append("/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0544/CBCT/20230510/CBCT_axial/dicom")

    # for f in error_folders:
    #     print(f)
    # dicom2nifti.dicom_series_to_nifti(
    #     str("/mnt/4TB1/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom"),
    #     str("/mnt/4TB1/onj/dataset/v0/Non_ONJ/EW-0429/CBCT/20200616/CBCT_axial/nifti/output.nii.gz"),
    # )


"""
Errors
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0429/CBCT/20200616/CBCT_axial/dicom
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0068/CBCT/20160811/CBCT_coronal/dicom
/mnt/aix22301/onj/dataset/v0/Non_ONJ_soi/EW-0135/CBCT/20220216/CBCT_sagittal/dicom # ignore sagittal
"""

# Non_ONJ_soi:
# EW-0544
# EW-0302
# EW-0543