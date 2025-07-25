import sys
import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def get_config_path() -> str:
    """Get config file path from command line arguments"""
    if len(sys.argv) != 2:
        print("Usage: python script.py <config.yaml>")
        print("Example: python train.py configs/train_dit_imagenet.yaml")
        sys.exit(1)
    
    return sys.argv[1]


def get_train_config() -> Dict[str, Any]:
    """Get training configuration with defaults"""
    
    # Default configuration
    default_config = {
        "module": "vit",
        "dataset": "cifar100",
        "checkpoint": None,
        "batch_size": 8,
        "max_epochs": 20,
        "accumulate_grad_batches": 4,
        "precision": "bf16",
        "project": "vit",
        "learning_rate": 1e-3,
        "reg_scale": 0.5,
        "quant_type": "none",
        "quant_rank": 192,
        "save_dir": "checkpoints",
        "devices": "1",
        "accelerator": "gpu",
        "strategy": "auto",
        "enable_progress_bar": False
    }
    
    # Load from config file
    config_path = get_config_path()
    file_config = load_config(config_path)
    default_config.update(file_config)
    
    return default_config


def get_val_config() -> Dict[str, Any]:
    """Get validation configuration with defaults"""
    
    # Default configuration
    default_config = {
        "module": "dit",
        "dataset": "imagenet", 
        "checkpoint": None,
        "batch_size": 8,
        "precision": 32,
        "learning_rate": 1e-6,
        "reg_scale": 0.5,
        "quant_type": "none", 
        "quant_rank": 192,
        "devices": "1",
        "accelerator": "gpu",
        "test": False
    }
    
    # Load from config file
    config_path = get_config_path()
    file_config = load_config(config_path)
    default_config.update(file_config)
                
    return default_config


class Config:
    """Configuration class to hold all settings"""
    def __init__(self, config_dict: Dict[str, Any]):
        for key, value in config_dict.items():
            setattr(self, key, value)
    
    def __repr__(self):
        items = [f"{k}={v}" for k, v in self.__dict__.items()]
        return f"Config({', '.join(items)})" 