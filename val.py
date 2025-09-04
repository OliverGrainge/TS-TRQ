import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv

from config_utils import Config, get_val_config

load_dotenv()
torch.set_float32_matmul_precision("high")


def get_module(config, kwargs, checkpoint_path=None):
    """Load model from checkpoint or create fresh model"""
    if config.module == "dit":
        from models import DiTModule

        if checkpoint_path is not None:
            return DiTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return DiTModule(**kwargs)
    elif config.module == "mlp":
        from models import MLPModule

        if checkpoint_path is not None:
            return MLPModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return MLPModule(**kwargs)
    elif config.module == "vit":
        from models import ViTModule

        if checkpoint_path is not None:
            return ViTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return ViTModule(**kwargs)
    else:
        raise ValueError(f"Invalid module: {config.module}")


def get_datamodule(config, kwargs):
    """Get the appropriate data module"""
    if config.dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**kwargs)
    elif config.dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**kwargs)


def main():
    # Load configuration from YAML file
    config_dict = get_val_config()
    config = Config(config_dict)

    print(f"Validation configuration: {config}")

    module_kwargs = {
        "learning_rate": config.learning_rate,
        "reg_scale": config.reg_scale,
    }

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

    # Load the model module (from checkpoint or create fresh)
    module = get_module(config, module_kwargs, checkpoint_path)

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
