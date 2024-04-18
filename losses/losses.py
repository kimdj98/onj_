import torch
from torch import nn

class BinaryCrossEntropy(nn.Module):
    def __init__(self):
        super(BinaryCrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        return -y_true * torch.log(y_pred) - (1 - y_true) * torch.log(1 - y_pred)

class CrossEntropy(nn.Module):
    def __init__(self):
        super(CrossEntropy, self).__init__()

    def forward(self, y_true, y_pred):
        return -torch.sum(y_true * torch.log(y_pred), dim=1)