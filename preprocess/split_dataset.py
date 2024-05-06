import hydra
from omegaconf import DictConfig
from pathlib import Path
from sklearn.model_selection import train_test_split
from typing import Tuple, List


def write_file(file: str, X: List[str], y: List[int]) -> None:
    """
    Write X and y to file in format: X(patient) y(onj_label)
    """
    with open(file, "w") as f:
        for i in range(len(X)):
            f.write(X[i].parent.name + "/" + X[i].name + " " + str(y[i]) + "\n")


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def split_dataset(cfg: DictConfig) -> None:
    """
    split dataset into train, val(, test) and write to file
    """
    split = cfg.data.split_ratio
    test_split = cfg.data.test_split
    data_dir = Path(cfg.data.data_dir)
    print(f"Spliiting dataset in {data_dir}... with split ratio: {split} and test split: {test_split}")

    onjs = list(data_dir.glob("ONJ_labeling/*"))
    non_onjs = list(data_dir.glob("Non_ONJ/*"))

    X = onjs + non_onjs

    y = [1] * len(onjs) + [0] * len(non_onjs)

    # split train, val(, test)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    if test_split:
        X_val, X_test, y_val, y_test = train_test_split(X_val, y_val, test_size=0.5, random_state=42)

    # write train, val(, test)
    write_file(data_dir / "train.txt", X_train, y_train)
    write_file(data_dir / "val.txt", X_val, y_val)
    if test_split:
        write_file(data_dir / "test.txt", X_test, y_test)


if __name__ == "__main__":
    split_dataset()
