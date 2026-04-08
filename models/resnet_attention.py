# models/resnet_attention.py

import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------
# Non-Local Attention Module
# ----------------------------------
class NonLocalBlock(nn.Module):

    def __init__(self, in_channels):
        super(NonLocalBlock, self).__init__()

        self.inter_channels = in_channels // 2

        self.theta = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.phi = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)
        self.g = nn.Conv2d(in_channels, self.inter_channels, kernel_size=1)

        self.out = nn.Conv2d(self.inter_channels, in_channels, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):

        batch_size, C, H, W = x.size()

        theta = self.theta(x).view(batch_size, self.inter_channels, -1)
        phi = self.phi(x).view(batch_size, self.inter_channels, -1)
        g = self.g(x).view(batch_size, self.inter_channels, -1)

        theta = theta.permute(0, 2, 1)

        attention = torch.bmm(theta, phi)

        attention = self.softmax(attention)

        g = g.permute(0, 2, 1)

        y = torch.bmm(attention, g)

        y = y.permute(0, 2, 1).contiguous()

        y = y.view(batch_size, self.inter_channels, H, W)

        y = self.out(y)

        return x + y


# ----------------------------------
# ResNet101 + Non-Local Attention
# ----------------------------------
class ResNet101_NonLocal(nn.Module):

    def __init__(self, num_classes=2):

        super(ResNet101_NonLocal, self).__init__()

        resnet = models.resnet101(weights=models.ResNet101_Weights.DEFAULT)

        self.conv1 = resnet.conv1
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool

        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4

        # Insert Non-Local Attention modules
        self.non_local3 = NonLocalBlock(1024)
        self.non_local4 = NonLocalBlock(2048)

        self.avgpool = resnet.avgpool

        self.fc = nn.Linear(2048, num_classes)


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)

        x = self.layer3(x)
        x = self.non_local3(x)

        x = self.layer4(x)
        x = self.non_local4(x)

        x = self.avgpool(x)

        x = torch.flatten(x, 1)

        x = self.fc(x)

        return x