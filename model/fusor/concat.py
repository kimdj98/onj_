# file description: This file contains the code for the Concatenation model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatModel(nn.Module):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module):
        super(ConcatModel, self).__init__()
        self.model_2d = model_2d
        self.model_3d = model_3d
        self.fc = nn.Linear(512, 2)

    def forward(self, x):
        o1 = self.model_2d(x)
        o2 = self.model_3d(x)
        return self.fc(torch.cat((o1, o2), dim=1))
