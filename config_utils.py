import sys
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    return config


def get_config_path() -> str:
    """Get config file path from command line arguments"""
    if len(sys.argv) != 2:
        print("Usage: python script.py <config.yaml>")
        print("Example: python train.py configs/train_dit_imagenet.yaml")
        sys.exit(1)

    return sys.argv[1]


def get_default_train_config() -> Dict[str, Any]:
    """Get default training configuration with structured sections"""

    return {
        # Module configuration
        "module_config": {
            "module": "vit",
            "learning_rate": 1e-3,
            "reg_scale": 0.5,
            "model_name": "Ahmed9275/Vit-Cifar100",
            "num_classes": 100,
            "checkpoint": None,
        },
        # Data module configuration
        "datamodule_config": {
            "dataset": "cifar100",
            "batch_size": 8,
            "num_workers": 4,
            "image_size": 224,
            "data_dir": "./data/raw",
            "download": True,
        },
        # Training configuration
        "train_config": {
            "max_epochs": 20,
            "accumulate_grad_batches": 4,
            "precision": "bf16",
            "accelerator": "gpu",
            "devices": "1",
            "strategy": "auto",
            "enable_progress_bar": False,
        },
        # Model checkpoint configuration
        "model_checkpoint_config": {
            "save_dir": "checkpoints",
            "monitor": "val_loss",
            "filename": "{epoch:02d}-{val_loss:.2f}",
            "save_top_k": 3,
            "mode": "min",
            "save_last": True,
        },
        # Logger configuration
        "logger_config": {"type": "wandb", "project": "vit", "log_model": False},
        # Quantization configuration
        "quantization_config": {"quant_type": "none", "rank": 192},
    }


def merge_configs(
    default_config: Dict[str, Any], file_config: Dict[str, Any]
) -> Dict[str, Any]:
    """Merge file configuration with default configuration, preserving structure"""
    merged_config = default_config.copy()

    for section_key, section_value in file_config.items():
        if section_key in merged_config and isinstance(section_value, dict):
            # Merge nested dictionary sections
            merged_config[section_key].update(section_value)
        else:
            # Direct replacement for non-dict values or new keys
            merged_config[section_key] = section_value

    return merged_config


def get_train_config() -> Dict[str, Any]:
    """Get training configuration with structured sections and defaults"""

    # Get default configuration
    default_config = get_default_train_config()

    # Load from config file
    config_path = get_config_path()
    file_config = load_config(config_path)

    # Merge configurations
    merged_config = merge_configs(default_config, file_config)

    return merged_config


def get_val_config() -> Dict[str, Any]:
    """Get validation configuration with structured sections"""

    # Default validation configuration
    default_config = {
        # Module configuration
        "module_config": {
            "module": "dit",
            "learning_rate": 1e-6,
            "reg_scale": 0.5,
            "checkpoint": None,  # Required for validation
        },
        # Data module configuration
        "datamodule_config": {"dataset": "imagenet", "batch_size": 8, "num_workers": 4},
        # Training configuration (validation specific)
        "train_config": {"precision": 32, "accelerator": "gpu", "devices": "1"},
        # Quantization configuration
        "quantization_config": {"quant_type": "none", "rank": 192},
        # Validation specific
        "test": False,
    }

    # Load from config file
    config_path = get_config_path()
    file_config = load_config(config_path)

    # Merge configurations
    merged_config = merge_configs(default_config, file_config)

    return merged_config


class Config:
    """Configuration class to hold all settings"""

    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)

    def __repr__(self):
        items = []
        for k, v in self.__dict__.items():
            if isinstance(v, dict):
                # Format nested dictionaries nicely
                nested_items = [f"{nk}={nv}" for nk, nv in v.items()]
                items.append(f"{k}={{{', '.join(nested_items)}}}")
            else:
                items.append(f"{k}={v}")
        return f"Config({', '.join(items)})"
