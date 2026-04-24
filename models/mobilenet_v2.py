# models/mobilenet_v2.py

import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------
# MobileNetV2 (Fine-tuned for dataset)
# ----------------------------------
class MobileNetV2_Model(nn.Module):

    def __init__(self, num_classes=2):
        super(MobileNetV2_Model, self).__init__()

        mobilenet = models.mobilenet_v2(
            weights=models.MobileNet_V2_Weights.DEFAULT
        )

        # Feature extractor
        self.features = mobilenet.features

        # Freeze early layers
        for param in self.features[:10].parameters():
            param.requires_grad = False

        # Train deeper layers
        for param in self.features[10:].parameters():
            param.requires_grad = True

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(1280, 512),
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