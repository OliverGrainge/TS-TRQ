import argparse
import os
from pathlib import Path

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv

load_dotenv()
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(description="Validate model on dataset")
    parser.add_argument(
        "--module",
        type=str,
        default="dit",
        choices=["dit", "mlp", "vit"],
        help="choose the trainer type",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet",
        choices=["imagenet", "cifar100"],
        help="choose the dataset",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="path to checkpoint to validate (optional - creates fresh model if not provided)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for validation"
    )
    parser.add_argument(
        "--precision", type=int, default=32, help="Precision (16, 32, etc.)"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-6,
        help="Learning rate (for fresh model)",
    )
    parser.add_argument(
        "--reg_scale",
        type=float,
        default=0.5,
        help="Regularization scale (for fresh model)",
    )
    parser.add_argument(
        "--quant_type",
        type=str,
        default="none",
        choices=["tsvdlinear", "trqlinear", "nbitlinear", "none"],
        help="Quantization type",
    )
    parser.add_argument("--quant_rank", type=int, default=192, help="Quantization rank")
    parser.add_argument(
        "--devices",
        type=str,
        default="1",
        help="Number of GPUs or comma separated list of GPU ids (e.g., '0,1')",
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="gpu",
        help="Validation accelerator: 'cpu', 'gpu', etc.",
    )
    parser.add_argument(
        "--test", action="store_true", help="Run on test set instead of validation set"
    )
    return parser.parse_args()


def get_module(args, kwargs, checkpoint_path=None):
    """Load model from checkpoint or create fresh model"""
    if args.module == "dit":
        from models import DiTModule

        if checkpoint_path is not None:
            return DiTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return DiTModule(**kwargs)
    elif args.module == "mlp":
        from models import MLPModule

        if checkpoint_path is not None:
            return MLPModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return MLPModule(**kwargs)
    elif args.module == "vit":
        from models import ViTModule

        if checkpoint_path is not None:
            return ViTModule.load_from_checkpoint(checkpoint_path, **kwargs)
        else:
            return ViTModule(**kwargs)
    else:
        raise ValueError(f"Invalid module: {args.module}")


def get_datamodule(args, kwargs):
    """Get the appropriate data module"""
    if args.dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**kwargs)
    elif args.dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**kwargs)


def main():
    args = parse_args()

    module_kwargs = {"learning_rate": args.learning_rate, "reg_scale": args.reg_scale}

    # Check if checkpoint exists (if provided)
    checkpoint_path = None
    if args.checkpoint is not None:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        print(f"Loading checkpoint: {checkpoint_path}")
        checkpoint_path = str(checkpoint_path)
    else:
        print("Creating fresh model for validation")

    # Load the model module (from checkpoint or create fresh)
    module = get_module(args, module_kwargs, checkpoint_path)

    # Apply quantization if specified
    if args.quant_type != "none":
        quant_kwargs = {"rank": args.quant_rank}
        module.apply_quantization(quant_type=args.quant_type, quant_kwargs=quant_kwargs)
        print(f"Applied {args.quant_type} quantization with rank {args.quant_rank}")

    # Load the Data Module
    datamodule_kwargs = {"batch_size": args.batch_size}
    datamodule = get_datamodule(args, datamodule_kwargs)
    datamodule.setup()

    # Get appropriate dataloader
    if args.test:
        dataloader = datamodule.test_dataloader()
        print("Running validation on test set")
    else:
        dataloader = datamodule.val_dataloader()
        print("Running validation on validation set")

    # Set up trainer for validation
    trainer = pl.Trainer(
        logger=False,  # No logging needed for validation
        precision=args.precision,
        accelerator=args.accelerator,
        devices=args.devices,
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
