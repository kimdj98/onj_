import torch
import torch.nn as nn
import torch.nn.functional as F

import hydra
from omegaconf import DictConfig


class FeatureExpand(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureExpand, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)
