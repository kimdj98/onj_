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


class ConcatModel2(nn.Module):
    def __init__(self, model_2d: nn.Module, model_3d: nn.Module, input_size: int = 512, num_classes: int = 2):
        super(ConcatModel, self).__init__()
        self.model_2d = model_2d
        self.model_3d = model_3d

        self.fc1 = nn.Linear(input_size, 1024)
        self.relu = nn.ReLU()

        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, batch):
        B, _, _, _ = batch["PA_image"].shape
        self.model_2d(batch["PA_image"])
        self.model_3d(batch["CT_image"])
        f2 = self.model_2d.hf.view(B, -1)
        f3 = self.model_3d.f.view(B, -1)

        # Concatenate the features
        concatenated_features = torch.cat((f2, f3), dim=1)

        # Pass through the additional fully connected layer with ReLU activation
        x = self.fc1(concatenated_features)
        x = self.relu(x)

        # Pass through the final fully connected layer for classification
        output = self.fc2(x)
        return output
