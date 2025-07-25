import argparse
import os

import pytorch_lightning as pl
import torch
from dotenv import load_dotenv
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger

load_dotenv()
torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train DiT on ImageNet with PyTorch Lightning"
    )
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
        help="choose the model name",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None, help="choose a checkpoint to load from"
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for training"
    )
    parser.add_argument(
        "--max_epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        type=int,
        default=4,
        help="Gradient accumulation steps",
    )
    parser.add_argument(
        "--precision", type=int, default=32, help="Precision (16, 32, etc.)"
    )
    parser.add_argument(
        "--project", type=str, default="dit-imagenet", help="wandb project name"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-6, help="Learning rate"
    )
    parser.add_argument(
        "--reg_scale", type=float, default=0.5, help="Regularization scale"
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
        "--save_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
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
        help="Training accelerator: 'cpu', 'gpu', etc.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="auto",
        help="Training strategy: 'ddp', 'dp', etc.",
    )
    return parser.parse_args()


def get_module(args, kwargs, checkpoint_path):
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
    if args.dataset == "imagenet":
        from data import ImageNetDataModule

        return ImageNetDataModule(**kwargs)
    elif args.dataset == "cifar100":
        from data import CIFAR100DataModule

        return CIFAR100DataModule(**kwargs)


def main():
    args = parse_args()
    module_kwargs = {"learning_rate": args.learning_rate, "reg_scale": args.reg_scale}
    quant_kwargs = {
        "rank": args.quant_rank,
    }
    datamodule_kwargs = {"batch_size": args.batch_size}

    # Load the model module
    module = get_module(args, module_kwargs, args.checkpoint)

    # Quantize the module
    if args.quant_type != "none":
        module.apply_quantization(quant_type=args.quant_type, quant_kwargs=quant_kwargs)

    print(module)

    # Load the Data Module
    datamodule = get_datamodule(args, datamodule_kwargs)

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    wandb_logger = WandbLogger(
        project=args.project + "_" + args.module, log_model=False
    )

    # Use pathlib for easier and more robust path handling
    from pathlib import Path

    # Determine checkpoint directory
    checkpoint_dir = Path(args.save_dir) / args.module / args.quant_type

    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=str(checkpoint_dir),
        filename="{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,
        mode="min",
        save_last=True,
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        log_every_n_steps=5,
        precision=args.precision,
        accumulate_grad_batches=args.accumulate_grad_batches,
        accelerator=args.accelerator,
        devices=args.devices,
        strategy=args.strategy,
        gradient_clip_val=1.0,
    )
    trainer.fit(module, train_loader, val_loader)


if __name__ == "__main__":
    main()
