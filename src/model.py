from torchvision.models import resnet18, ResNet18_Weights
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights
import torch.nn as nn

class FaceClassifier(nn.Module):
    def __init__(self, backbone: str = "resnet18"):
        super().__init__()
        if backbone == "resnet18":
            self.backbone = resnet18(weights=ResNet18_Weights.DEFAULT)
            num_feats = self.backbone.fc.in_features
            # Remove original FC layer
            self.backbone.fc = nn.Identity()
        elif backbone == "mobilenet_v2":
            self.backbone = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
            num_feats = self.backbone.classifier[1].in_features
            # Remove original classifier
            self.backbone.classifier = nn.Identity()
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Add a simple 2‐class head
        self.classifier = nn.Sequential(
            nn.Linear(num_feats, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        feats = self.backbone(x)
        logits = self.classifier(feats)
        return logits
