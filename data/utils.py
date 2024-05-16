from enum import Enum


class Modal(Enum):
    MDCT = "MDCT"
    CBCT = "CBCT"
    PA = "PA"
    BoneSPECT = "BoneSPECT"
    ClinicalData = "ClinicalData"


class Direction(Enum):
    AXIAL = "axial"
    SAGITTAL = "sagittal"
    CORONAL = "coronal"
