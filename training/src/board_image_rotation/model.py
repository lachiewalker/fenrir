import torch
import torch.nn as nn

from src import consts, common
import src.board_image_rotation.dataset as dataset

from torchvision import models


class ImageRotation(nn.Module):
    def __init__(self, backbone: str = "mobilenet_v3_large"):
        super(ImageRotation, self).__init__()
        self.backbone = backbone
        if backbone == "regnet_x_800mf":
            self.model = models.regnet_x_800mf(
                weights=models.RegNet_X_800MF_Weights.IMAGENET1K_V2
            )
            self.model.fc = torch.nn.LazyLinear(out_features=len(dataset.ROTATIONS))
        elif backbone == "mobilenet_v3_large":
            self.model = models.mobilenet_v3_large(
                weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2
            )
            self.model.classifier[-1] = nn.Linear(1280, len(dataset.ROTATIONS))
        else:
            raise ValueError(f"Unknown backbone {backbone}")

    def forward(self, x):
        return self.model(x)
