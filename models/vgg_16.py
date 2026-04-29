# models/vgg_16.py

import torch
import torch.nn as nn
import torchvision.models as models

# ----------------------------------
# VGG16 (Training From Scratch Baseline)
# ----------------------------------
class VGG16_Model(nn.Module):

    def __init__(self, num_classes=2):
        super(VGG16_Model, self).__init__()

        # weights=None completely initializes the model with random numbers
        vgg = models.vgg16(weights=None)

        # Feature extractor
        self.features = vgg.features

        # Pooling layer
        self.avgpool = vgg.avgpool

        # Memory Compressed Fully Connected Layer (so your 4GB RTX 3050 doesn't crash)
        self.fc = nn.Sequential(
            nn.Linear(512 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):

        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x