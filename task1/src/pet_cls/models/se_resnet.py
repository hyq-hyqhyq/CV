from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.resnet import BasicBlock, ResNet


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, hidden, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, channels, _, _ = x.shape
        weights = self.pool(x).view(batch_size, channels)
        weights = self.fc(weights).view(batch_size, channels, 1, 1)
        return x * weights


class SEBasicBlock(BasicBlock):
    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: nn.Module | None = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: type[nn.Module] | None = None,
    ) -> None:
        super().__init__(
            inplanes=inplanes,
            planes=planes,
            stride=stride,
            downsample=downsample,
            groups=groups,
            base_width=base_width,
            dilation=dilation,
            norm_layer=norm_layer,
        )
        self.se = SEBlock(planes * self.expansion)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)
        return out


def se_resnet18(num_classes: int = 37, dropout: float = 0.0) -> ResNet:
    model = ResNet(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes)
    if dropout > 0:
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
    return model


def load_pretrained_resnet18_backbone(model: nn.Module) -> dict[str, Any]:
    reference = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    pretrained_state = reference.state_dict()
    current_state = model.state_dict()
    compatible = {
        key: value
        for key, value in pretrained_state.items()
        if key in current_state and current_state[key].shape == value.shape
    }
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    return {"loaded_keys": len(compatible), "missing_keys": missing, "unexpected_keys": unexpected}
