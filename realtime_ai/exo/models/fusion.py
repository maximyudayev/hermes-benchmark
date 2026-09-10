"""
Filename: hermes/revalexo/ai/utils/models/fusion.py
Author: Diwas Lamsal <diwaslamsal123@hotmail.com>
Date: 2026-03-06
Version: 1.0
Description: Definitions of modality fusion models.
"""

import torch
import torch.nn as nn
from typing import Dict, List
from .base_models import BaseEncoder, MultiHorizonClassifier


class FusionModel(nn.Module):
    """Fusion model combining features from multiple modality encoders.

    Uses concat + LayerNorm fusion.
    """

    def __init__(
        self,
        modality_encoders: Dict[str, BaseEncoder],
        fusion_method: str = "concat_layernorm",
        hidden_dim: int = 512,
        num_classes: int = 13,
        dropout: float = 0.5,
        prediction_horizons: List[float] = [0],
        shared_classifier_layers: bool = True,
        **kwargs,
    ):
        super().__init__()

        self.modality_encoders = nn.ModuleDict()
        self.modalities = []
        self.feature_dims = {}
        self.fusion_method = fusion_method
        self.prediction_horizons = prediction_horizons
        self.num_prediction_heads = len(prediction_horizons)
        self.normalize_features = True

        # Process encoders
        for modality, encoder in modality_encoders.items():
            self.modality_encoders[modality] = encoder
            self.modalities.append(modality)
            self.feature_dims[modality] = encoder.get_feature_dim()

        # Feature normalization per modality
        self.modality_norms = nn.ModuleDict()
        for modality, encoder in modality_encoders.items():
            feature_dim = encoder.get_feature_dim()
            self.modality_norms[modality] = nn.BatchNorm1d(feature_dim)

        # Concat + LayerNorm fusion
        if fusion_method == "concat_layernorm":
            total_dim = sum(self.feature_dims.values())
            self.concat_ln = nn.LayerNorm(total_dim)
            self.fusion = nn.Sequential(nn.Linear(total_dim, hidden_dim), nn.ReLU())
        else:
            raise ValueError(
                f"Unsupported fusion method: {fusion_method}. "
                f"This realtime module only supports 'concat_layernorm'."
            )

        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=hidden_dim,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            shared_layers=shared_classifier_layers,
        )

    def forward(self, **inputs):
        """Forward pass with modality inputs as keyword arguments.

        Args:
            **inputs: e.g. raw_imu=tensor, video=tensor, image=tensor.

        Returns:
            List[torch.Tensor]: Classification outputs for each prediction horizon.
        """

        features = {}

        # TODO: launch modality encoders in parallel if possible.
        # with cuda.stream(inputs[modality][0]):

        for modality in self.modality_encoders.keys():
            if modality in inputs and inputs[modality] is not None:
                encoder: BaseEncoder = self.modality_encoders[modality]
                feat = encoder.extract_features(inputs[modality])
                if self.normalize_features:
                    feat = self.modality_norms[modality](feat)
                features[modality] = feat

        if not features:
            raise ValueError("No valid inputs were provided for any modality")

        # TODO: gather and fuse features before classification head.
        # if self._device.type == "cuda":
        #     cuda.synchronize(self._device)

        # Concat + LayerNorm
        combined = torch.cat(list(features.values()), dim=1)
        combined = self.concat_ln(combined)
        x = self.fusion(combined)

        outputs = self.classifier(x)
        return outputs

    def get_prediction_horizons(self) -> List[float]:
        return self.prediction_horizons.copy()

    def get_num_prediction_heads(self) -> int:
        return self.num_prediction_heads

    def get_feature_dim(self) -> int:
        if hasattr(self.fusion, "__getitem__"):
            first_layer = self.fusion[0]
            if hasattr(first_layer, "out_features"):
                return first_layer.out_features
        return 512
