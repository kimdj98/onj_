import sys
import pathlib

sys.path.append(".")

import hydra
from data.onj_dataset import get_data_loader
from model.detection.model import TempModel
from tqdm import tqdm


@hydra.main(config_path="../config", config_name="config")
def main(conf):
    data_dir = pathlib.Path(conf.data_dir) / conf.data_version  # path to /dataset

    train_loader = get_data_loader(conf, "train")

    model = TempModel(conf)

    for p_folder, _, label in train_loader:
        dir = data_dir / p_folder[0]

        if p_folder[0].startswith("ONJ"):
            # print(p_folder, label)
            pass

        elif p_folder[0].startswith("Non_ONJ"):
            # print(p_folder, label)
            pass

        else:
            raise ValueError(f"Invalid patient folder name: {p_folder[0]}")


if __name__ == "__main__":
    main()
