from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional


class ModalityType(Enum):
    UNKNOWN = "unknown"
    MULTIMODAL = "multimodal"
    RAW_IMU = "raw_imu"
    IMAGE = "image"
    VIDEO = "video"


class ModelType(Enum):
    RAW_IMU = ModalityType.RAW_IMU.value
    IMAGE = ModalityType.IMAGE.value
    VIDEO = ModalityType.VIDEO.value
    FUSION = "fusion"


@dataclass
class ModalityConfig:
    type: ModalityType
    receptive_field: Optional[int] = 120
    stride: Optional[int] = 1
    column_patterns: List[str] = field(default_factory=lambda: ["*acc_*", "*gyro_*"])
    eval_transforms: List = field(default_factory=[])


@dataclass
class ModelConfig:
    type: ModelType
    name: str
    params: Dict


@dataclass
class Config:
    prediction_horizons: List[float]
    window_size: float
    num_classes: int
    device: str
    checkpoint_path: str
    label_mapping: Dict[str, int]
    predictions_bit_width: int
    modalities: Dict[ModalityType, ModalityConfig]
    models: Dict[ModelType, ModelConfig]
