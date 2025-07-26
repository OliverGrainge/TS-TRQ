import os

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pathlib import Path
import sys 
from typing import Dict, Any
import yaml
from config_utils import get_train_config, Config

load_dotenv()
torch.set_float32_matmul_precision("high")

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    
    return config



def get_module(module_config):
    module_type = module_config.pop("module", "vit")
    
    if module_type == "dit":
        from models import DiTModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            return DiTModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            return DiTModule(**module_config)
    elif module_type == "mlp":
        from models import MLPModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            return MLPModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            return MLPModule(**module_config)
    elif module_type == "vit":
        from models import ViTModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            return ViTModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            print(module_config)
            return ViTModule(**module_config)
    else:
        raise ValueError(f"Invalid module: {module_type}")


def get_datamodule(datamodule_config):
    dataset = datamodule_config.pop("dataset", "cifar100")
    
    if dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**datamodule_config)
    elif dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**datamodule_config)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def create_model_checkpoint_callback(model_checkpoint_config):
    """Create ModelCheckpoint callback from configuration"""
    from pathlib import Path
    
    # Set defaults if not provided
    config = {
        "monitor": "val_loss",
        "filename": "{epoch:02d}-{val_loss:.2f}",
        "save_top_k": 3,
        "mode": "min",
        "save_last": True,
        **model_checkpoint_config
    }
    
    # Handle dirpath creation
    if "save_dir" in config:
        dirpath = Path(config.pop("save_dir"))
        config["dirpath"] = str(dirpath)
    
    return ModelCheckpoint(**config)


def create_logger(logger_config):
    """Create logger from configuration"""
    logger_type = logger_config.get("type", "wandb")
    
    if logger_type == "wandb":
        return WandbLogger(**{k: v for k, v in logger_config.items() if k != "type"})
    else:
        raise ValueError(f"Unsupported logger type: {logger_type}")


def main():
    # Load configuration from YAML file
    config_path = sys.argv[1]
    config_dict = load_config(config_path)
    config = Config(config_dict)
    
    print(f"Training configuration: {config}")
    
    # Extract configuration sections
    module_config = getattr(config, 'module_config', {})
    datamodule_config = getattr(config, 'datamodule_config', {})
    train_config = getattr(config, 'train_config', {})
    model_checkpoint_config = getattr(config, 'model_checkpoint_config', {})
    logger_config = getattr(config, 'logger_config', {})
    quantization_config = getattr(config, 'quantization_config', {})
    
    # Load the model module
    module = get_module(module_config)

    # Apply quantization if specified
    quant_type = quantization_config.pop("quant_type", "none")
    if quant_type != "none":
        module.apply_quantization(quant_type=quant_type, **quantization_config)

    print(module)

    # Load the Data Module
    datamodule = get_datamodule(datamodule_config)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    # Create logger
    logger = create_logger(logger_config)

    # Create model checkpoint callback
    checkpoint_callback = create_model_checkpoint_callback(model_checkpoint_config)

    # Create trainer with train_config parameters
    trainer_kwargs = {
        "callbacks": [checkpoint_callback],
        "logger": logger,
        "log_every_n_steps": 5,
        "gradient_clip_val": 1.0,
        **train_config
    }
    
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
