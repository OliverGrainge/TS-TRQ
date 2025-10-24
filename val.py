import os
import sys
from pathlib import Path
from typing import Any, Dict

import pytorch_lightning as pl
import code
import torch
import yaml
from dotenv import load_dotenv

load_dotenv()
torch.set_float32_matmul_precision("high")


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
        print("Usage: python val.py <config.yaml>")
        print("Example: python val.py runs/configs/test/cifar10-unet-diffusion/tsvd-reg[0.5]-rank[32]-A[fp].yaml")
        sys.exit(1)

    return sys.argv[1]


def get_val_config() -> Dict[str, Any]:
    """Load configuration directly from YAML file"""
    config_path = get_config_path()
    return load_config(config_path)


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


def get_diffusion_module(kwargs, checkpoint_path=None):
    """Load model from checkpoint or create fresh model"""
    from models import DiffusionModule

    if checkpoint_path is not None:
        print("==============" * 10, kwargs)
        module = DiffusionModule(**kwargs)
        return module.load_from_checkpoint(checkpoint_path)
    else:
        return DiffusionModule(**kwargs)


def get_datamodule(config, kwargs):
    """Get the appropriate data module"""
    if config.dataset == "cifar10":
        from data import CIFAR10DataModule

        return CIFAR10DataModule(**kwargs)
    else: 
        raise ValueError(f"Invalid dataset: {config.dataset}")


def main():
    # Load configuration from YAML file
    config_dict = get_val_config()
    config = Config(config_dict)

    module_kwargs = {key: value for key, value in config.module_config.items() if key != "module_type" and key != "module"}

    print(f"Validation configuration: {config}")

    # Check if checkpoint exists (if provided)
    checkpoint_path = None
    if config.checkpoint is not None:
        checkpoint_path = Path(config.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    else:
        print("Creating fresh model for validation")

      # Drop into an interactive REPL here

    # Load the model module (from checkpoint or create fresh)
    module = get_diffusion_module(module_kwargs, checkpoint_path)
    code.interact(local=dict(globals(), **locals()))
    

    # Apply quantization if specified
    if config.quant_type != "none":
        quant_kwargs = {"rank": config.quant_rank}
        module.apply_quantization(
            quant_type=config.quant_type, quant_kwargs=quant_kwargs
        )
        print(f"Applied {config.quant_type} quantization with rank {config.quant_rank}")

    # Load the Data Module
    datamodule_kwargs = {"batch_size": config.batch_size}
    datamodule = get_datamodule(config, datamodule_kwargs)
    datamodule.setup()

    # Get appropriate dataloader
    if config.test:
        dataloader = datamodule.test_dataloader()
        print("Running validation on test set")
    else:
        dataloader = datamodule.val_dataloader()
        print("Running validation on validation set")

    # Set up trainer for validation
    trainer = pl.Trainer(
        logger=False,  # No logging needed for validation
        precision=config.precision,
        accelerator=config.accelerator,
        devices=config.devices,
        enable_checkpointing=False,  # No checkpointing needed
        enable_progress_bar=True,
    )

    # Run validation
    print("Starting validation...")
    results = trainer.validate(module, dataloader, verbose=True)

    # Print results
    if results:
        print("\nValidation Results:")
        for key, value in results[0].items():
            print(f"{key}: {value:.6f}")

    print("Validation completed!")


if __name__ == "__main__":
    main()
