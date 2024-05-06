# cls_pa_dataset.py
# Description: This file contains the distribution of pa images to {base_dir}/CLS_PA -> ({base_dir}/CLS_PA/Non_ONJ, {base_dir}/CLS_PA/ONJ).

import hydra
from omegaconf import DictConfig
from pathlib import Path
import os
from tqdm import tqdm


def copy_PA(f: Path, t: Path) -> None:  # f: from, t: to
    for patient in tqdm(f.iterdir()):
        if not (patient / "panorama").exists():
            continue

        if len(os.listdir(patient / "panorama")) == 0:
            continue

        import shutil
        import glob

        source_files = glob.glob(f"{patient}/panorama/*.jpg")
        if len(source_files) == 1:
            shutil.copy(source_files[0], f"{t}/{patient.name}.jpg")
        else:
            print(f"Expected 1 file, found {len(source_files)}: {source_files}")


@hydra.main(version_base="1.1", config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    BASE_PATH = Path(cfg.data.data_dir)
    DATA_PATH = "CLS_PA"

    if not os.path.exists(f"{BASE_PATH}/{DATA_PATH}"):
        os.makedirs(f"{BASE_PATH}/{DATA_PATH}/ONJ")
        os.makedirs(f"{BASE_PATH}/{DATA_PATH}/Non_ONJ")

    copy_PA(BASE_PATH / "ONJ_labeling", BASE_PATH / "CLS_PA" / "ONJ")
    copy_PA(BASE_PATH / "Non_ONJ_soi", BASE_PATH / "CLS_PA" / "Non_ONJ")


if __name__ == "__main__":
    main()
