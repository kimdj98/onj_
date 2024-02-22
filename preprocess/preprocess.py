import hydra
import os

from colorama import Fore, Style

from enum import Enum
import cv2
import numpy as np
import matplotlib.pyplot as plt
from omegaconf import DictConfig, OmegaConf
from sklearn.model_selection import train_test_split
from tqdm import tqdm


from pathlib import Path
from typing import Tuple, List


class DIRECTIONS(Enum):
    """
    Enum class for directions of image
    """

    AXIAL = "axial"
    CORONAL = "coronal"
    SAGITTAL = "sagittal"


class MODALS(Enum):
    """
    Enum class for modalities of ONJ
    """

    CBCT = "CBCT"
    MDCT = "MDCT"
    BoneSPECT = "BoneSPECT"
    BoneSCAN = "BoneSCAN"
    ClinicalPicture = "ClinicalPicture"
    Clinical = "Clinical"
    panorama = "panorama"
    record = "record"


class ImgProcessor:
    def __init__(self, conf: DictConfig) -> None:
        self.conf = conf.preprocess.img_process

    def process(self, im: np.ndarray, modality: str, conf: DictConfig = None) -> np.ndarray:
        if conf is None:
            conf = self.conf

        if modality.name == "CBCT":
            # date -> direction -> extension -> file
            return cv2.resize(im, conf.CT.size)

        elif modality.name == "MDCT":
            # date -> direction -> extension -> file
            return cv2.resize(im, conf.CT.size)

        elif modality.name == "BoneSPECT":
            # date -> direction -> extension -> file
            pass

        elif modality.name == "BoneSCAN":
            # date -> direction -> wholebodyscan -> extension -> file
            pass

        elif modality.name == "panorama":
            # date.jpg
            return cv2.resize(im, conf.panorama.size)

        else:
            raise ValueError(f"{modality.name} is not a valid modality.")


def write_file(file: str, X: List[str], y: List[int]) -> None:
    """
    Write X and y to file in format: X(patient) y(onj_label)
    """
    with open(file, "w") as f:
        for i in range(len(X)):
            f.write(X[i].parent.name + "/" + X[i].name + " " + str(y[i]) + "\n")


class PreProcessor:
    def __init__(self, conf: DictConfig) -> None:
        self.data_dir = Path(conf.preprocess.data_dir)

        onjs = list(self.data_dir.glob("ONJ/*"))  # chain generator and make it to list
        non_onjs = list(self.data_dir.glob("Non_ONJ/*"))
        self.patients = onjs + non_onjs
        self.img_processor = ImgProcessor(conf)

        # Mapping of non-standard names to standard names
        self.standardization_map = {
            "Bone SCAN": "BoneSCAN",
            "Bonescan": "BoneSCAN",
            "Bonespect": "BoneSPECT",
            "임상사진": "ClinicalPicture",
            # TODO: Add other mappings if necessary
        }

        self.modalities = [item.value for item in MODALS]  # list of modalities

    def rename(self) -> None:
        """
        Rename non-standard folder names to standard names
        """
        for patient in self.patients:
            for folder in patient.glob("*"):
                if folder.name in self.standardization_map:
                    os.rename(patient / folder, patient / self.standardization_map[folder.name])

    def split(self, ratio: float, test_split: bool = False) -> None:
        onjs = list(self.data_dir.glob("ONJ/*"))
        non_onjs = list(self.data_dir.glob("Non_ONJ/*"))
        X = onjs + non_onjs

        # TODO: later subdivide label to 0 ~ 4 (0: Non-ONJ, 1: ONJ-stage1, 2: ONJ-stage2, 3: ONJ-stage3, 4: ONJ-stage4)
        y = [1] * len(onjs) + [0] * len(non_onjs)

        # split train, val(, test)
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
        if test_split:
            X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

        # write train, val(, test)
        write_file(self.data_dir / "train.txt", X_train, y_train)
        write_file(self.data_dir / "val.txt", X_val, y_val)
        if test_split:
            write_file(self.data_dir / "test.txt", X_test, y_test)

    def plot_hist(self) -> None:
        """
        Plot histogram of which modality is used frequently
        """
        raise NotImplementedError

    def write_file(self, image, dest_dir) -> None:
        """
        Write an image to the specified directory. This method is intended to be used
        as a helper function by the 'write_images' and 'preprocess' methods.

        Args:
            image: The image to be written.
            dest_dir: The destination directory where the image will be saved.
        """
        dest_dir.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dest_dir), image)

    def write_images(self, images, modality) -> None:
        """
        Process and write a collection of images to files. This method is typically
        called by the 'preprocess' method.

        Args:
            images: A collection of image paths to be processed and written.
            modality: The modality of the images, used in processing.
        """
        for im in images:
            image = cv2.imread(str(im))
            image = self.img_processor.process(image, modality)
            self.write_file(image, dest_dir=im.parents[6] / "preprocessed" / im.relative_to(*im.parts[:6]))

    def preprocess(self) -> None:
        """
        Preprocess images and save them to preprocessed folder
        """
        dest_dir = self.data_dir / "preprocessed"
        if dest_dir.exists():
            print(f"{Fore.RED}Error: {dest_dir} already exists.{Style.RESET_ALL}")
            return

        for patient in tqdm(self.patients):
            modalities = list(patient.glob("*"))
            for modality in modalities:
                if modality.name not in self.modalities:
                    if "임상정보" in modality.name:
                        continue
                    print(f"{modality.name} is not a valid modality.")
                    print(f"modaility path: {modality}")
                    continue

                if modality.name == "CBCT":
                    # date -> direction -> extension -> file
                    for direction in DIRECTIONS:
                        images = modality.glob(f"*/{modality.name}_{direction.value}/jpg/*")
                        self.write_images(images, modality)

                elif modality.name == "MDCT":
                    # date -> direction -> extension -> file
                    for direction in DIRECTIONS:
                        images = modality.glob(f"*/{modality.name}_{direction.value}/jpg/*")
                        self.write_images(images, modality)

                elif modality.name == "panorama":
                    # date.jpg
                    images = modality.glob("*.jpg")
                    self.write_images(images, modality)

                # elif modality.name == "BoneSPECT":
                #     # date -> direction -> extension -> file
                #     self.img_processor.process(modality.name, (512, 512))

                # elif modality.name == "BoneSCAN":
                #     # date -> direction -> wholebodyscan -> extension -> file
                #     self.img_processor.process(modality.name, (512, 512))

                else:
                    continue


@hydra.main(config_path="../config", config_name="config")
def main(conf: DictConfig) -> None:
    preprocessor = PreProcessor(conf)

    # rename non-standard folder names to standard names
    # preprocessor.rename()

    # creates metadata train.txt, val.txt, test.txt
    # preprocessor.split(conf.preprocess.split_ratio, test_split=True)

    preprocessor.preprocess()
    # print(conf.data_dir)


if __name__ == "__main__":
    main()



"""
functions to add.
1. rename non-standard folder names to standard names
2. split train, val(, test)
3. plot histogram of which modality is used frequently
4. preprocess images and save them to preprocessed folder
5. distribute labels to each patient modal directions directory
"""