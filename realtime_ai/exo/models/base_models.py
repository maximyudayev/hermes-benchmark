import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List


class BaseEncoder(nn.Module, ABC):
    """
    Base class for all encoder models.
    Defines the interface for compatibility with multimodal fusion.
    """

    def __init__(self, feature_dim: int, **kwargs):
        super().__init__()
        self.feature_dim = feature_dim
        self.prediction_horizons = kwargs.get("prediction_horizons", [0])
        self.num_prediction_heads = len(self.prediction_horizons)

    @abstractmethod
    def encode_features(self, x: torch.Tensor) -> torch.Tensor:
        """Extract features from input without classification head."""
        pass

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        """Public method to extract features."""
        return self.encode_features(x)

    def get_feature_dim(self) -> int:
        return self.feature_dim

    def get_prediction_horizons(self) -> List[float]:
        return self.prediction_horizons.copy()

    def get_num_prediction_heads(self) -> int:
        return self.num_prediction_heads


class MultiHorizonClassifier(nn.Module):
    """
    Multi-horizon classifier that creates separate heads for each prediction horizon.
    """

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        prediction_horizons: List[float],
        dropout: float = 0.5,
        shared_layers: bool = True,
    ):
        super().__init__()

        self.prediction_horizons = prediction_horizons
        self.num_heads = len(prediction_horizons)
        self.num_classes = num_classes
        self.shared_layers = shared_layers

        if shared_layers and self.num_heads > 1:
            self.shared_net = nn.Sequential(
                nn.Linear(input_dim, input_dim), nn.ReLU(), nn.Dropout(dropout)
            )
            self.heads = nn.ModuleList(
                [nn.Linear(input_dim, num_classes) for _ in range(self.num_heads)]
            )
        else:
            self.shared_net = nn.Identity()
            self.heads = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Dropout(dropout), nn.Linear(input_dim, num_classes)
                    )
                    for _ in range(self.num_heads)
                ]
            )

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        shared_features = self.shared_net(x)
        outputs = []
        for head in self.heads:
            outputs.append(head(shared_features))
        return outputs
