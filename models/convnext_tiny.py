# models/convnext_tiny.py

import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------
# ConvNeXt Tiny (Fine-tuned for dataset)
# ----------------------------------
class ConvNeXt_Tiny_Model(nn.Module):

    def __init__(self, num_classes=2):
        super(ConvNeXt_Tiny_Model, self).__init__()

        convnext = models.convnext_tiny(
            weights=models.ConvNeXt_Tiny_Weights.DEFAULT
        )

        # Feature extractor
        self.features = convnext.features

        # Freeze early layers
        for param in self.features[:4].parameters():
            param.requires_grad = False

        # Train deeper layers
        for param in self.features[4:].parameters():
            param.requires_grad = True

        # Pooling layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(768, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )


    def forward(self, x):

        x = self.features(x)

        x = self.avgpool(x)

        x = self.fc(x)

        return x