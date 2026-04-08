# models/vgg16_model.py

import torch
import torch.nn as nn
import torchvision.models as models


# ----------------------------------
# VGG16 (Fine-tuned for 13k dataset)
# ----------------------------------
class VGG16_Model(nn.Module):

    def __init__(self, num_classes=2):
        super(VGG16_Model, self).__init__()

        vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)

        # Feature extractor (like conv layers in ResNet)
        self.features = vgg.features

        # Freeze early layers
        for param in self.features[:20].parameters():
            param.requires_grad = False

        # Train deeper layers
        for param in self.features[20:].parameters():
            param.requires_grad = True

        # Pooling layer
        self.avgpool = vgg.avgpool

        # Fully connected layer (like fc in ResNet)
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