import torch.nn as nn
from torchvision import models

class FaceClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.backbone = models.resnet18(pretrained=True)
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)
