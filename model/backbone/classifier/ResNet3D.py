import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock3D(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, num_blocks, num_classes=2):
        super(ResNet3D, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv3d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm3d(64)

        self.fm1 = None  # fm: feature map
        self.fm2 = None
        self.fm3 = None
        self.fm4 = None
        self.f = None  # f: last feature
        self.out = None  # out: output

        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer1.register_forward_hook(self.hook_fn1)

        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer2.register_forward_hook(self.hook_fn2)

        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer3.register_forward_hook(self.hook_fn3)

        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.layer4.register_forward_hook(self.hook_fn4)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.avgpool.register_forward_hook(self.hook_fn5)

        self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.fc.register_forward_hook(self.hook_fn6)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def hook_fn1(self, module, input, output):
        self.fm1 = output

    def hook_fn2(self, module, input, output):
        self.fm2 = output

    def hook_fn3(self, module, input, output):
        self.fm3 = output

    def hook_fn4(self, module, input, output):
        self.fm4 = output

    def hook_fn5(self, module, input, output):
        self.f = output

    def hook_fn6(self, module, input, output):
        self.out = output

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class Bottleneck3D(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck3D, self).__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


def resnet18_3d():
    return ResNet3D(BasicBlock3D, [2, 2, 2, 2])


def resnet34_3d():
    return ResNet3D(BasicBlock3D, [3, 4, 6, 3])


def resnet50_3d():
    return ResNet3D(Bottleneck3D, [2, 2, 2, 2])


def resnet101_3d():
    return ResNet3D(Bottleneck3D, [3, 4, 23, 3])


if __name__ == "__main__":
    # Initialize the model
    model = resnet101_3d().cuda()

    # Print the model summary (Requires torchsummary)
    from torchsummary import summary

    summary(model, (1, 64, 512, 512))
