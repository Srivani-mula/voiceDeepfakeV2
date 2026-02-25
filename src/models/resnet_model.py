import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Replace final FC layer
        self.model.fc = nn.Linear(self.model.fc.in_features, num_classes)

    def forward(self, x):
        """
        Expected input shape:
        (batch_size, 3, 128, 200)
        """

        # Resize to 224x224 for ResNet
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False)

        x = self.model(x)

        return x