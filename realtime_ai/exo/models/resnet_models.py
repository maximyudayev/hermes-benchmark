"""
Filename: hermes/revalexo/ai/utils/models/resnet_models.py
Author: Diwas Lamsal <diwaslamsal123@hotmail.com>
Date: 2026-03-06
Version: 1.0
Description: Model definitions of ResNet-based image backbones.
"""

import torch.nn as nn
import torchvision.models as models
from .base_models import BaseEncoder, MultiHorizonClassifier


class ResNet_Image(BaseEncoder):
    """Image model using pretrained ResNet backbone from torchvision.

    Supports ResNet-18.
    """

    def __init__(
        self,
        num_classes=11,
        pretrained=True,
        model_size="18",
        feature_dim=None,
        freeze_backbone=False,
        freeze_early_layers=False,
        prediction_horizons=[0],
        shared_classifier_layers=True,
    ):
        super().__init__(
            feature_dim=feature_dim or 512, prediction_horizons=prediction_horizons
        )

        model_dict = {
            "18": models.resnet18,
        }

        if model_size not in model_dict:
            raise ValueError(
                f"Invalid ResNet size. Choose from: {list(model_dict.keys())}"
            )

        weights = "DEFAULT" if pretrained else None
        self.backbone = model_dict[model_size](weights=weights)

        self.backbone_dim = self.backbone.fc.in_features

        # Remove classification head
        self.backbone.fc = nn.Identity()

        # Freeze parameters if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        elif freeze_early_layers:
            for name, param in self.backbone.named_parameters():
                if any(
                    layer in name
                    for layer in ["conv1", "bn1", "layer1", "layer2", "layer3"]
                ):
                    param.requires_grad = False

        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self.feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=self.feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.5,
            shared_layers=shared_classifier_layers,
        )

    def encode_features(self, x):
        """Extract features without classification head.

        Args:
            x: Input tensor [B, 3, H, W].

        Returns:
            Feature tensor [B, feature_dim].
        """
        backbone_features = self.backbone(x)
        features = self.feature_projector(backbone_features)
        return features

    def forward(self, x):
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs


class ResNet18_Image(ResNet_Image):
    """ResNet-18 for image classification."""

    def __init__(self, **kwargs):
        super().__init__(model_size="18", **kwargs)
