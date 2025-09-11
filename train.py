import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import yaml
from dotenv import load_dotenv

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

from config_utils import Config

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


def flatten_dict(d, parent_key="", sep="/"):
    """Flatten a nested dictionary for logging purposes"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def print_header(title: str, width: int = 80):
    """Print a formatted header"""
    print("\n" + "=" * width)
    print(f" {title} ".center(width))
    print("=" * width)


def print_section(title: str, width: int = 80):
    """Print a formatted section header"""
    print(f"\n{title}")
    print("-" * len(title))


def format_config_value(value):
    """Format configuration values for display"""
    if isinstance(value, dict):
        return f"Dict({len(value)} items)"
    elif isinstance(value, list):
        return f"List({len(value)} items)"
    elif isinstance(value, (str, int, float, bool)):
        return str(value)
    else:
        return str(type(value).__name__)


def print_config(config: Config):
    """Print configuration in a nicely formatted way"""
    print_header("TRAINING CONFIGURATION")

    # Get all configuration sections
    sections = [
        ("Module Config", getattr(config, "module_config", {})),
        ("Data Config", getattr(config, "datamodule_config", {})),
        ("Training Config", getattr(config, "train_config", {})),
        ("Checkpoint Config", getattr(config, "model_checkpoint_config", {})),
        ("Logger Config", getattr(config, "logger_config", {})),
        ("Quantization Config", getattr(config, "quantization_config", {})),
    ]

    for section_name, section_config in sections:
        if section_config:  # Only print non-empty sections
            print_section(section_name)
            for key, value in section_config.items():
                formatted_value = format_config_value(value)
                print(f"  {key:25} : {formatted_value}")


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    """Count total and trainable parameters in a model"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_params, trainable_params


def format_parameter_count(count: int) -> str:
    """Format parameter count with appropriate units"""
    if count >= 1e9:
        return f"{count/1e9:.2f}B"
    elif count >= 1e6:
        return f"{count/1e6:.2f}M"
    elif count >= 1e3:
        return f"{count/1e3:.2f}K"
    else:
        return str(count)


def print_model_info(module: pl.LightningModule):
    """Print model information in a nicely formatted way"""
    print_header("MODEL INFORMATION")

    # Basic model info
    print_section("Model Details")
    print(f"  Model Type        : {type(module).__name__}")

    # Parameter counts
    total_params, trainable_params = count_parameters(module)
    print(
        f"  Total Parameters  : {format_parameter_count(total_params)} ({total_params:,})"
    )
    print(
        f"  Trainable Params  : {format_parameter_count(trainable_params)} ({trainable_params:,})"
    )

    # Device and dtype info
    device = (
        next(module.parameters()).device
        if len(list(module.parameters())) > 0
        else "N/A"
    )
    dtype = (
        next(module.parameters()).dtype if len(list(module.parameters())) > 0 else "N/A"
    )
    print(f"  Device            : {device}")
    print(f"  Data Type         : {dtype}")

    # Model-specific info
    if hasattr(module, "is_quantized"):
        print(f"  Quantized         : {module.is_quantized}")
        if module.is_quantized and hasattr(module, "quant_type"):
            print(f"  Quantization Type : {module.quant_type}")

    if hasattr(module, "pretrained"):
        print(f"  Pretrained        : {module.pretrained}")

    if hasattr(module, "image_size"):
        print(f"  Image Size        : {module.image_size}")

    if hasattr(module, "learning_rate"):
        print(f"  Learning Rate     : {module.learning_rate}")

    # Print model architecture
    print_section("Model Architecture")
    print(module)


def print_training_start_info(datamodule, logger):
    """Print information about the training setup"""
    datamodule.setup()
    print_header("TRAINING SETUP")

    # Dataset info
    print_section("Dataset Information")
    if hasattr(datamodule, "num_classes"):
        print(f"  Number of Classes : {datamodule.num_classes}")

    # Try to get dataset sizes
    try:
        datamodule.setup()
        train_size = (
            len(datamodule.train_dataloader().dataset)
            if hasattr(datamodule.train_dataloader(), "dataset")
            else "Unknown"
        )
        batch_size = (
            datamodule.train_dataloader().batch_size
            if hasattr(datamodule.train_dataloader(), "batch_size")
            else "Unknown"
        )
        print(f"  Training Samples  : {train_size}")
        print(f"  Batch Size        : {batch_size}")

        if train_size != "Unknown" and batch_size != "Unknown":
            steps_per_epoch = train_size // batch_size
            print(f"  Steps per Epoch   : {steps_per_epoch}")
    except:
        print(f"  Training Samples  : Unknown")

    # Logger info
    print_section("Logging Setup")
    logger_type = type(logger).__name__
    print(f"  Logger Type       : {logger_type}")

    if hasattr(logger, "save_dir"):
        print(f"  Save Directory    : {logger.save_dir}")
    if hasattr(logger, "name"):
        print(f"  Experiment Name   : {logger.name}")
    if hasattr(logger, "project"):
        print(f"  Project           : {logger.project}")


def get_module(module_config):
    """Load and return the specified model module"""
    module_type = module_config.pop("module_type", "vit")

    if module_type == "resnet":
        from models import ResNetModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            print(f"="*100)
            print(f"Loading ResNetModule from checkpoint: {checkpoint}")
            print(f"="*100)
            return ResNetModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            return ResNetModule(**module_config)
    elif module_type == "stablediffusion":
        from models import StableDiffusionModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            print(f"="*100)
            print(f"Loading stablediffusion from checkpoint: {checkpoint}")
            print(f"="*100)
            return StableDiffusionModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            return StableDiffusionModule(**module_config)
    elif module_type == "diffusion":
        from models import DiffusionModule

        checkpoint = module_config.pop("checkpoint", None)
        if checkpoint is not None:
            print(f"="*100)
            print(f"Loading diffusion from checkpoint: {checkpoint}")
            print(f"="*100)
            return DiffusionModule.load_from_checkpoint(checkpoint, **module_config)
        else:
            return DiffusionModule(**module_config)
    else:
        raise ValueError(f"Invalid module: {module_type}")


def get_datamodule(datamodule_config):
    """Load and return the specified data module"""
    dataset = datamodule_config.pop("dataset", "cifar100")

    if dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**datamodule_config)
    elif dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**datamodule_config)
    elif dataset == "cifar10":
        from data import CIFAR10DataModule

        return CIFAR10DataModule(**datamodule_config)
    else:
        raise ValueError(f"Invalid dataset: {dataset}")


def create_model_checkpoint_callback(model_checkpoint_config):
    """Create ModelCheckpoint callback from configuration"""
    from pathlib import Path

    filename_prefix = (
        model_checkpoint_config.pop("filename_prefix", "")
        if "filename_prefix" in model_checkpoint_config
        else ""
    )

    # Set defaults if not provided
    config = {
        "monitor": "val_loss",
        "filename": f"{filename_prefix}{{epoch:02d}}-{{val_loss:.2f}}",
        "save_top_k": 2,
        "mode": "min",
        **model_checkpoint_config,
    }

    return ModelCheckpoint(**config)


def create_logger(logger_config, config_dict=None):
    """Create logger from configuration

    Supports both WandB and TensorBoard loggers based on configuration.

    Args:
        logger_config: Dictionary containing logger configuration
        config_dict: Optional full configuration to log as hyperparameters

    Returns:
        Configured logger instance

    Example logger_config:
        # For WandB
        {
            "type": "wandb",
            "project": "my-project",
            "name": "experiment-1"
        }

        # For TensorBoard
        {
            "type": "tensorboard",
            "save_dir": "./logs",
            "name": "experiment-1"
        }
    """
    logger_type = logger_config.get("type", "wandb").lower()

    if logger_type == "wandb":
        logger_kwargs = {k: v for k, v in logger_config.items() if k != "type"}
        logger = WandbLogger(**logger_kwargs)

        # Log the full configuration as hyperparameters
        if config_dict is not None:
            logger.log_hyperparams(flatten_dict(config_dict))

        return logger

    else:
        raise ValueError(
            f"Unsupported logger type: {logger_type}. Supported types: 'wandb'"
        )


def main():
    """Main training function"""
    # Load configuration from YAML file
    if len(sys.argv) < 2:
        raise ValueError("Please provide a configuration file path as argument")

    config_path = sys.argv[1]
    config_dict = load_config(config_path)
    config = Config(config_dict)

    # Print nicely formatted configuration
    print_config(config)

    # Extract configuration sections
    module_config = getattr(config, "module_config", {})
    datamodule_config = getattr(config, "datamodule_config", {})
    train_config = getattr(config, "train_config", {})
    model_checkpoint_config = getattr(config, "model_checkpoint_config", {})
    logger_config = getattr(config, "logger_config", {})
    quantization_config = getattr(config, "quantization_config", {})

    # Load the model module
    module = get_module(module_config)

    # Apply quantization if specified
    quant_type = quantization_config.pop("quant_type", None)
    if quant_type != None:
        module.apply_quantization(quant_type=quant_type, **quantization_config)

    # Print nicely formatted model information
    print_model_info(module)

    # Load the Data Module
    datamodule = get_datamodule(datamodule_config)

    # Create logger
    logger = create_logger(logger_config, config_dict)

    # Create model checkpoint callback
    checkpoint_callback = create_model_checkpoint_callback(model_checkpoint_config)

    # Print training setup information
    print_training_start_info(datamodule, logger)

    # Create trainer with train_config parameters
    trainer_kwargs = {
        "callbacks": [checkpoint_callback],
        "logger": logger,
        "log_every_n_steps": 5,
        "gradient_clip_val": 1.0,
        **train_config,
    }

    print_header("STARTING TRAINING")
    trainer = pl.Trainer(**trainer_kwargs)
    trainer.fit(module, datamodule)


if __name__ == "__main__":
    main()
