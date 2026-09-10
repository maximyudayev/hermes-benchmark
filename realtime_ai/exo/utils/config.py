import json
import os
import torch
import yaml
from math import ceil, log2

from .types import Config, ModalityConfig, ModalityType, ModelConfig, ModelType


def load_config(
    config_path: str,
    device: torch.device,
    checkpoint_path: str | None = None,
) -> Config:
    """Load live inference configuration from supplied YAML."""
    config_path = os.path.abspath(config_path)
    config_dir = os.path.dirname(config_path)

    with open(config_path, "r") as f:
        config_file: dict = yaml.safe_load(f)

    # Checkpoint
    if checkpoint_path:
        checkpoint_path = os.path.abspath(checkpoint_path)
    elif config_file.get("checkpoint"):
        ckpt = config_file["checkpoint"]
        if not os.path.isabs(ckpt):
            ckpt = os.path.join(config_dir, ckpt)
        checkpoint_path = os.path.abspath(ckpt)

    # Modalities
    modalities: dict[ModalityType, ModalityConfig] = {}
    for key, mod_raw in config_file["modalities"].items():
        modality_type = ModalityType(key)
        modalities[modality_type] = ModalityConfig(modality_type, **mod_raw)

    # Models
    models: dict[ModelType, ModelConfig] = {}
    for key, model_raw in config_file["models"].items():
        model_type = ModelType(key)
        models[model_type] = ModelConfig(model_type, **model_raw)

    # Label mapping
    label_file = os.path.join(config_dir, "label_mapping.json")
    with open(label_file, "r") as f:
        label_mapping = json.load(f)
        num_classes = len(label_mapping.get("idx_to_label", {}))

    predictions_bit_width = 8 * 2 ** ceil(log2((num_classes.bit_length() + 7) // 8))

    # Top-level wrapper
    config = Config(
        prediction_horizons=config_file["prediction_horizons"],
        window_size=config_file["window_size"],
        num_classes=num_classes,
        device=device,
        checkpoint_path=checkpoint_path,
        label_mapping=label_mapping,
        predictions_bit_width=predictions_bit_width,
        modalities=modalities,
        models=models,
    )

    return config


def get_modality_type(config: Config) -> ModalityType:
    """Determine modality type."""
    mods = list(config.modalities.keys())
    if len(mods) == 1:
        return mods[0]
    elif len(mods) > 1:
        return ModalityType.MULTIMODAL
    else:
        return ModalityType.UNKNOWN
