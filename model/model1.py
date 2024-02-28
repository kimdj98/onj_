import hydra

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import sys

sys.path.append(".")

from data.onj_dataset import ONJDataset, get_data_loader

from typing import Tuple, List, Dict, Union, Any
from omegaconf import DictConfig, OmegaConf


class TempModel(nn.Module):
    def __init__(self, conf: DictConfig) -> None:
        super().__init__()
        self.conf = conf.model

    def forward(self, x: Tensor) -> Tensor:
        # TODO: Implement forward pass
        return x


@hydra.main(version_base="1.3", config_path="../config", config_name="config")
def main(conf):
    train_loader = get_data_loader(conf, "train")
    for i, batch in enumerate(train_loader):
        p_folder, annotations, label = batch
        if p_folder[0] == "ONJ":
            print(p_folder, label)
            break
        else:
            continue

    val_loader = get_data_loader(conf, "val")
    for i, batch in enumerate(val_loader):
        p_folder, annotations, label = batch
        if p_folder[0] == "ONJ":
            print(p_folder, label)
            break
        else:
            continue

    test_loader = get_data_loader(conf, "test")
    for i, batch in enumerate(test_loader):
        p_folder, annotations, label = batch
        if p_folder[0] == "ONJ":
            print(p_folder, label)
            break
        else:
            continue


if __name__ == "__main__":
    main()
