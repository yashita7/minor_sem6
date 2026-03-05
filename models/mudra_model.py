import timm
import torch.nn as nn

class MudraNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.net = timm.create_model(
            "efficientnetv2_b2",
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.net(x)
