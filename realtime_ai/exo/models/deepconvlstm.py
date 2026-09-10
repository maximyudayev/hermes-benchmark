import torch.nn as nn
from .base_models import BaseEncoder, MultiHorizonClassifier


class DeepConvLSTM(BaseEncoder):
    """
    DeepConvLSTM model for IMU-based activity recognition.
    Based on Ordonez and Roggen (https://www.mdpi.com/1424-8220/16/1/115).
    """

    def __init__(
        self,
        channels,
        num_classes,
        window_size,
        conv_kernels=64,
        conv_kernel_size=5,
        lstm_units=128,
        lstm_layers=2,
        dropout=0.5,
        feature_dim=None,
        prediction_horizons=[0],
        shared_classifier_layers=True,
    ):
        feature_dim = feature_dim or lstm_units
        super(DeepConvLSTM, self).__init__(
            feature_dim=feature_dim, prediction_horizons=prediction_horizons
        )

        self.channels = channels
        self.num_classes = num_classes
        self.conv_kernels = conv_kernels
        self.lstm_units = lstm_units

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, conv_kernels, (conv_kernel_size, 1))
        self.conv2 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv3 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))
        self.conv4 = nn.Conv2d(conv_kernels, conv_kernels, (conv_kernel_size, 1))

        # LSTM layer
        self.lstm = nn.LSTM(
            channels * conv_kernels,
            lstm_units,
            num_layers=lstm_layers,
            batch_first=True,
        )

        # Dropout
        self.dropout = nn.Dropout(dropout)

        # Multi-horizon classification head
        self.classifier = MultiHorizonClassifier(
            input_dim=lstm_units,
            num_classes=num_classes,
            prediction_horizons=prediction_horizons,
            dropout=dropout,
            shared_layers=shared_classifier_layers,
        )

        self.activation = nn.ReLU()
        self.final_seq_len = window_size - (conv_kernel_size - 1) * 4

        # Feature projection if needed for multimodal setting
        if feature_dim != lstm_units:
            self.feature_projector = nn.Linear(lstm_units, feature_dim)
        else:
            self.feature_projector = nn.Identity()

    def encode_features(self, x):
        """
        Extract features without classification head.

        Args:
            x: Input tensor [batch_size, window_size, channels]

        Returns:
            Feature tensor [batch_size, feature_dim]
        """
        x = x.unsqueeze(1)  # Add channel dimension for 2D convolution

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.activation(self.conv3(x))
        x = self.activation(self.conv4(x))

        # Prepare for LSTM
        x = x.permute(0, 2, 1, 3)
        x = x.reshape(x.shape[0], x.shape[1], -1)

        x, h = self.lstm(x)

        # Get last time step output
        features = x[:, -1, :]

        return self.feature_projector(features)

    def forward(self, x):
        """
        Forward pass.

        Args:
            x: Input tensor [batch_size, window_size, channels]

        Returns:
            List[torch.Tensor]: Classification outputs for each prediction horizon
        """
        features = self.encode_features(x)
        outputs = self.classifier(features)
        return outputs
