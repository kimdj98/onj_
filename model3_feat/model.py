# combine all the parts of the model2
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()

    def forward(self, x):
        return x
