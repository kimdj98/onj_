import torch
import json
from pathlib import Path


def normalize(json_file: Path) -> None:
    with open(json_file, "r") as file:
        labels = json.load(file)


if __name__ == "__main__":
    normalize(Path("../dataset/v0/label_v231122/label.json"))
