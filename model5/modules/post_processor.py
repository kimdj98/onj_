import torch.nn as nn
import torch

class ImageFeatureExtractor(nn.Module):
    def __init__(self, dim: int, seq_len: int, hidden_dim: int):
        super(ImageFeatureExtractor, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)
        self.fc3 = nn.Linear(seq_len, 1)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x).squeeze(-1)
        return x

class Classifier(nn.Module):
    def __init__(self, dim: int, hidden_dim: int = 512, n_class: int = 1):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_class)
        self.gelu = nn.GELU()

    def forward(self, x: torch.Tensor):
        x = self.fc1(x)
        x = self.gelu(x)
        x = self.fc2(x)
        return x
