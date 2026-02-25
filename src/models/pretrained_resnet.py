import torch
import torch.nn as nn
import torchvision.models as models

class PretrainedResNet18(nn.Module):
    def __init__(self, num_classes=2, freeze_backbone=True):
        super().__init__()

        self.backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

        # Change first conv to accept 1-channel log-mel
        self.backbone.conv1 = nn.Conv2d(
            1, 64, kernel_size=7, stride=2, padding=3, bias=False
        )

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Replace classifier
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
