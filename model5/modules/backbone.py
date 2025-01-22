import torch.nn as nn
import torch
import torch.nn.functional as F

from torchvision.models import resnet18
class BasicBlock_2D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock_2D, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_channels, out_channels, kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(out_channels),
            )

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(x)
        out = self.relu(out)
        return out


class ResNet18_2D(nn.Module):
    def __init__(self):
        super(ResNet18_2D, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(
            kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False
        )

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512, num_classes) # use only the backbone of ResNet18

    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        if stride != 1 or in_channels != out_channels:
            layers = []
            layers.append(BasicBlock_2D(in_channels, out_channels, stride=2))
            for _ in range(1, blocks):
                layers.append(BasicBlock_2D(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)

        else:
            layers = []
            layers.append(BasicBlock_2D(in_channels, out_channels, stride=1))
            for _ in range(1, blocks):
                layers.append(BasicBlock_2D(out_channels, out_channels, stride=1))
            return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # x = self.avgpool(x)
        # x = torch.flatten(x, 1)
        # x = self.fc(x)
        return x


if __name__ == "__main__":
    input = torch.randn(1, 3, 224, 224)
    model = ResNet18_2D(num_classes=2)
    output = model(input)
