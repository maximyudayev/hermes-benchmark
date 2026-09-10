"""
Filename: hermes/revalexo/ai/utils/models/video_models.py
Author: Diwas Lamsal <diwaslamsal123@hotmail.com>
Date: 2026-03-06
Version: 1.0
Description: Model definitions of video backbones.
"""

import torch
import torch.nn as nn
from .base_models import BaseEncoder, MultiHorizonClassifier


class X3D_Video(BaseEncoder):
    """Video model using X3D backbone from PyTorchVideo.

    Supports xs, s, m, l sizes.
    """

    def __init__(
        self,
        num_classes=13,
        pretrained=False,
        model_size="xs",
        feature_dim=None,
        freeze_backbone=False,
        prediction_horizons=[0],
        shared_classifier_layers=True,
    ):
        self.backbone_dim = None
        self._feature_dim = feature_dim or 512
        super().__init__(
            feature_dim=self._feature_dim, prediction_horizons=prediction_horizons
        )

        # Create X3D model
        model_name = f"x3d_{model_size}"
        self.backbone = torch.hub.load(
            "facebookresearch/pytorchvideo", model_name, pretrained=pretrained
        )

        # Freeze backbone if requested
        if freeze_backbone:
            for name, param in self.backbone.named_parameters():
                if not name.startswith("blocks.5"):
                    param.requires_grad = False

        last_block = self.backbone.blocks[-1]

        if hasattr(last_block, "proj"):
            self.backbone_dim = last_block.proj.in_features
            last_block.proj = nn.Identity()
        else:
            # Fallback: determine output dim via test forward pass
            with torch.no_grad():
                dummy_input = torch.zeros(1, 3, 16, 224, 224)
                features = self.backbone(dummy_input)
                self.backbone_dim = features.shape[1]

        # Feature projection
        self.feature_projector = nn.Sequential(
            nn.Linear(self.backbone_dim, self._feature_dim),
            nn.ReLU(True),
            nn.Dropout(0.5),
        )

        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=self._feature_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=0.5,
            shared_layers=shared_classifier_layers,
        )

    def get_feature_dim(self) -> int:
        """Return the dimension of extracted features (after projection)."""
        return self._feature_dim

    def extract_features(self, x):
        """Extract projected features. Used by FusionModel for feature fusion.

        Args:
            x: Video tensor [C, T, H, W].

        Returns:
            Feature tensor [1, feature_dim].
        """
        return self.encode_features(x)

    def encode_features(self, x):
        """Extract projected features (after `feature_projector`).

        Used by standalone forward pass.

        Args:
            x: Input tensor [C, T, H, W].

        Returns:
            Feature tensor [1, feature_dim].
        """
        backbone_features = self.backbone(x.unsqueeze(0))
        features = self.feature_projector(backbone_features)
        return features

    def forward(self, x):
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs
