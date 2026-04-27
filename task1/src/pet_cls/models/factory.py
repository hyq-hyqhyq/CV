from __future__ import annotations

from typing import Any

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18

from .cbam_resnet import cbam_resnet18 as _cbam_resnet18
from .cbam_resnet import load_pretrained_resnet18_backbone as _load_pretrained_cbam_backbone
from .se_resnet import load_pretrained_resnet18_backbone, se_resnet18


def _replace_classifier(model: nn.Module, num_classes: int, dropout: float) -> nn.Module:
    in_features = model.fc.in_features  # type: ignore[attr-defined]
    if dropout > 0:
        model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))  # type: ignore[attr-defined]
    else:
        model.fc = nn.Linear(in_features, num_classes)  # type: ignore[attr-defined]
    return model


def build_model(config: dict[str, Any], num_classes: int) -> tuple[nn.Module, dict[str, Any]]:
    model_cfg = config["model"]
    name = str(model_cfg["name"]).lower()
    pretrained = bool(model_cfg["pretrained"])
    dropout = float(model_cfg.get("dropout", 0.0))
    metadata: dict[str, Any] = {"model_name": name, "pretrained": pretrained}

    if name == "resnet18":
        weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        model = resnet18(weights=weights)
        model = _replace_classifier(model, num_classes=num_classes, dropout=dropout)
    elif name in {"se_resnet18", "seresnet18"}:
        model = se_resnet18(num_classes=num_classes, dropout=dropout)
        if pretrained:
            metadata["pretrained_backbone"] = load_pretrained_resnet18_backbone(model)
    elif name in {"cbam_resnet18", "cbamresnet18"}:
        model = _cbam_resnet18(num_classes=num_classes, dropout=dropout)
        if pretrained:
            metadata["pretrained_backbone"] = _load_pretrained_cbam_backbone(model)
    elif name in {"vit_tiny", "vit_tiny_patch16_224", "swin_t", "swin_tiny"}:
        try:
            import timm
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Missing dependency 'timm'. Install it (pip install timm) to use ViT/Swin models."
            ) from exc

        timm_name = {
            "vit_tiny": "vit_tiny_patch16_224",
            "vit_tiny_patch16_224": "vit_tiny_patch16_224",
            "swin_t": "swin_tiny_patch4_window7_224",
            "swin_tiny": "swin_tiny_patch4_window7_224",
        }[name]
        model = timm.create_model(timm_name, pretrained=pretrained, num_classes=num_classes)
    else:
        raise ValueError(f"Unsupported model name: {model_cfg['name']}")

    return model, metadata
