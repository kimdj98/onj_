# file description: This file contains the code for the Concatenation model

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConcatModel(nn.Module):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel, self).__init__()
        self.model_2d = model_2d
        self.model_3d = model_3d
        self.fc = nn.Linear(input_size, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.hf.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)
        return self.fc(torch.cat((f2, f3), dim=1))
